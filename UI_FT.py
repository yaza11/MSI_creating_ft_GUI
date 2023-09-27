from from_txt import msi_from_txt, get_ref_peaks, search_peak_th, create_feature_table

import matplotlib
import os
import re
import numpy as np
import pandas as pd
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from typing import Iterable
from dataclasses import dataclass
import sys
from PyQt5 import QtWidgets, uic, QtCore, QtGui
import qdarktheme


# Ensure using PyQt5 backend
matplotlib.use('Qt5Agg')

ui_file = r'MSI_creating_FT.ui'


def check_file_integrity(
        file: str, is_file: bool = True, suffixes: list[str] = None
) -> bool:
    """Check if a given file exists and optionally is of right type."""
    if os.path.exists(file):
        if is_file != os.path.isfile(file):
            print(f'{file} is not the right type (folder instead of file or vise versa)')
            return False
        elif is_file and (suffixes is not None):
            if (suffix := os.path.splitext(file)[1]) not in suffixes:
                print(f'{file} should type should be one of {suffixes}, not {suffix}')
                return False
            else:
                print(f'{file} is okay')
                return True
        else:
            print(f'{file} is okay')
            return True
    elif file != '':
        print(f'{file} does not exist.')
    return False


@dataclass
class Options:
    peak_th_ref_peaks: float
    norm_spectra: bool
    save_error_FT: bool


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass  # Implement the flush method as a no-op


class MplCanvas(FigureCanvas):

    def __init__(self):
        fig = plt.Figure()
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def replace_figure_and_axes(self, new_figure, new_axes):
        # Remove the current axes from the current figure
        self.figure.delaxes(self.axes)

        # Assign the new axes and figure
        self.figure = new_figure
        self.axes = new_axes

        # Add the new axes to the new figure
        self.figure.add_axes(self.axes)


class UI_FT(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI_FT, self).__init__()
        uic.loadUi(ui_file, self)
        self.initiate_plt_area()
        self.show()
        # console output in textView
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        # load default options
        self.update_options()

        # link buttons to functions
        self.link_widgets()

        self.thresholds = []  # threshold values explored
        self.params_thrs = {}  # n_ref, tic_coverage, mean_error and sparsity for each thr

    def closeEvent(self, event):
        # Restore sys.stdout when the GUI is closed
        sys.stdout = sys.__stdout__
        event.accept()

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.textEdit_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_console.setTextCursor(cursor)
        self.textEdit_console.ensureCursorVisible()

    def link_widgets(self):
        self.btn_read_data.clicked.connect(self.read_data)  # read txt file
        self.btn_browse_data.clicked.connect(self.get_txt_file_from_dialog)  # open dialog window to select txt file
        self.btn_browse_img_folder.clicked.connect(self.get_save_dir_from_dialog)  # open dialog window to select save folder
        self.btn_find_thrs.clicked.connect(self.search_peak_th_params)
        self.btn_save_plot.clicked.connect(self.save_table)
        self.btn_load.clicked.connect(self.load_settings)
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_stop.clicked.connect(self.set_stop)

    def read_data(self):
        """Activated by read-btn."""
        try:
            file = self.lineEdit_file_spectra.text()
            file = re.findall(r'(?:file:///)?(.+)', file)[0]
        except:
            return
        if not check_file_integrity(file, suffixes=['.txt']):
            return

        self.spectra = msi_from_txt(file)
        print('finished msi_from_txt')

    def get_txt_file_from_dialog(self):
        """Path to txt file with spectra to load in."""
        txt_file = QtWidgets.QFileDialog.getOpenFileName(self, 'File to read', 'c:\\', '*.txt')[0]
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_file_spectra')
        le.setText(txt_file)

    def get_save_dir_from_dialog(self):
        img_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Directory for saving feature table', 'c:\\')
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_dir_imgs')
        le.setText(img_dir)

    def get_mass_file_from_dialog(self):
        file = QtWidgets.QFileDialog.getOpenFileName(self, 'File for masses', 'c:\\', '(*.txt *.csv *.xlsx)')[0]
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_mass_list')

    def update_options(self):
        try:
            peak_th_ref_peaks = self.lineEdit_peak_th.text()
            peak_th_ref_peaks = [float(thr) for thr in peak_th_ref_peaks.split(';')]
        except:
            print('thresholds must be separated by semicolon and numbers between 0 and 1')
            return
        try:
            SMALL_SIZE = self.horizontalScrollBar.value() / 10
            MEDIUM_SIZE = SMALL_SIZE * 3 / 2
            BIGGER_SIZE = SMALL_SIZE * 5 / 3

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            # print('updated small fs to', SMALL_SIZE)
        except:
            pass

        norm_spectra = self.checkBox_norm_spectra.isChecked()
        save_error_FT = self.checkBox_save_errors.isChecked()

        self.opts = Options(
            peak_th_ref_peaks=peak_th_ref_peaks,
            norm_spectra=norm_spectra,
            save_error_FT=save_error_FT
        )

    def initiate_plt_area(self):
        placeholder = self.findChild(QtWidgets.QWidget, 'plt_area')

        # Get the existing layout of the widget or create a new one if it doesn't have a layout
        layout = QtWidgets.QVBoxLayout()
        placeholder.setLayout(layout)

        self.canvas = MplCanvas()

        # Add the FigureCanvas to the layout
        navigation_toolbar = NavigationToolbar(self.canvas, self)
        navigation_toolbar.setStyleSheet("background-color: white;")
        layout.addWidget(navigation_toolbar)
        layout.addWidget(self.canvas)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Set stretch factor to 1 to make it expand to fill the available space
        layout.setStretchFactor(self.canvas, 1)

        self.canvas.show()

    def set_stop(self):
        if not self.has_valid_mass_file:
            return
        print('stopping plotting')
        self.stop = True

    def search_peak_th_params(self):
        """
        Search peak threshold for msi data.

        Plot different parameters important for evaluating the peak threshold
        for creating the featuer table.

        Parameters
        ----------
        peak_th_ref_peaks : list[float]
            list of peak_th to check.

        Returns
        -------
        None.

        """
        if not hasattr(self, 'spectra'):
            print('read in a txt file with data first')
            return

        self.stop = False
        updated = False
        # get threshold vals and check their validity
        try:
            thrs = np.array(
                [float(thr) for thr in self.lineEdit_search_thr.text().split(';')]
            )
            if np.any(thrs >= 1) or np.any(thrs < 0):
                print('threshold values should be numbers between 0 and 1 and separated by semicolon')
                return
        except:
            print('threshold values should be numbers between 0 and 1 and separated by semicolon')
            return

        print(f'searching peaks for {thrs}')

        for thr in thrs:
            if self.stop:
                print('breaking loop')
                break

            self.fig = plt.Figure(
                figsize=self.canvas.figure.get_size_inches(),
                dpi=self.canvas.figure.get_dpi(),
                layout='constrained'
            )
            self.ax = self.fig.add_subplot()

            QtWidgets.QApplication.processEvents()
            if thr not in self.thresholds:
                self.thresholds.append(thr)
                out = search_peak_th(self.spectra.copy(), thr)
                out = {metric: value[0] for metric, value in out.items()}
                out['n_ref'] = len(out['n_ref'])
                self.params_thrs[thr] = out
            # plot result
            self.df_params = pd.DataFrame.from_dict(self.params_thrs).T
            self.df_params.sort_index(inplace=True)
            for metric in self.df_params.columns:
                values = self.df_params[metric].to_numpy()
                self.ax.plot(
                    self.df_params.index,
                    values / np.max(values),
                    '-o',
                    label=f'{metric}, max={np.max(values):.2f}'
                )
            self.ax.set_xlabel('peak th')
            self.ax.set_ylabel('metric relative to max')
            self.ax.legend()
            self.ax.set_title('Peak threshold metrics')
            self.update_plt_area()

        self.stop = False

    def update_plt_area(self):
        self.update_options()
        self.canvas.replace_figure_and_axes(self.fig, self.ax)
        self.canvas.draw()

    def save_table(self):
        self.update_options()
        try:
            thrs = [float(thr) for thr in self.lineEdit_peak_th.text().split(';')]
            if np.any(np.array(thrs) >= 1) or np.any(np.array(thrs) < 0):
                print('threshold values should be numbers between 0 and 1 and separated by semicolon')
                return
        except:
            print('threshold contains numbers that can not be interpreted as number')
            return
        if self.checkBox_norm_spectra.isChecked():
            norm_mode = 'median'
        else:
            norm_mode = 'None'
        if self.lineEdit_image_name.text() != '':
            name_ft = self.lineEdit_image_name.text()
        else:
            name_ft = 'feature_table'
        name_err_ft = name_ft + '_err'
        path_feature_table = self.lineEdit_dir_imgs.text()

        ref_peaks = get_ref_peaks(self.spectra.copy(), peak_th=thrs)

        for peak_th in thrs:
            print(f'saving feature table with peak threshold {peak_th}')
            name_suffix = f'_thr{str(peak_th).replace(".", "d")}.csv'
            name_ft += name_suffix
            name_err_ft += name_suffix
            feature_table, error_table = create_feature_table(
                self.spectra.copy(),
                ref_peaks[peak_th],
                normalization=norm_mode
            )

            feature_table = feature_table.sort_values(by=['y', 'x'])

            feature_table.to_csv(os.path.join(path_feature_table, name_ft))
            if self.checkBox_save_errors.isChecked():
                error_table.to_csv(os.path.join(path_feature_table, name_err_ft))
        print(f'saved feature table as {os.path.join(path_feature_table, name_ft)}')

    def save_settings(self):
        fields = [
            'lineEdit_file_spectra', 'lineEdit_search_thr', 'lineEdit_peak_th',
            'lineEdit_dir_imgs', 'lineEdit_image_name',
            'checkBox_norm_spectra', 'checkBox_save_errors',
            'horizontalScrollBar'
        ]

        entries = []
        for field in fields:
            widget_type = field.split('_')[0]
            if widget_type == 'lineEdit':
                e = self.findChild(QtWidgets.QLineEdit, field).text()
            elif widget_type == 'comboBox':
                e = self.findChild(QtWidgets.QComboBox, field).currentText()
            elif widget_type == 'checkBox':
                e = self.findChild(QtWidgets.QCheckBox, field).isChecked()
            elif widget_type == 'horizontalScrollBar':
                e = self.findChild(QtWidgets.QScrollBar, field).value()
            else:
                raise NotImplementedError
            entries.append(e)
        d = dict(zip(fields, entries))
        with open('gui_settings_FT.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('saved settings')

    def load_settings(self):
        print('loading settings')
        with open('gui_settings_FT.pickle', 'rb') as handle:
            d = pickle.load(handle)

        for field, entry in d.items():
            try:
                widget_type = field.split('_')[0]
                if widget_type == 'lineEdit':
                    e = self.findChild(QtWidgets.QLineEdit, field)
                    e.setText(entry)
                elif widget_type == 'comboBox':
                    e = self.findChild(QtWidgets.QComboBox, field)
                    e.setCurrentText(entry)
                elif widget_type == 'checkBox':
                    e = self.findChild(QtWidgets.QCheckBox, field)
                    e.setChecked(entry)
            except:
                pass

        self.read_data()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = UI_FT()
    sys.exit(app.exec_())
