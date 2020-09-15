import uuid
from PIL import Image
import subprocess
import psutil
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from UI_mainWin import Ui_MainWindow
import os
# pyuic5 UI_design/mainwindow.ui -o UI_mainWin.py


class mainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        self.main_win.setWindowTitle('Semantic Image Inpainting')
        logo_path = "/Users/Joe/Downloads/8f39b75f-4edb-4529-81d2-2ac7797e400f_200x200.png"
        self.ui.titlePic_label.setPixmap(QPixmap(logo_path))

        self.gif = "/Users/Joe/Downloads/ezgif.com-resize (1).gif"
        self.movie = QMovie(self.gif, QByteArray(), None)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.setSpeed(100)
        self.ui.loadgif_lbl.setMovie(self.movie)
        self.ui.loadgif_lbl.setHidden(True)
        self.enable_all_btns(True)
        self.ui.quality_better_rdBtn.setChecked(True)
        if self.ui.maskType_custom_rdBtn.isChecked():
            self.ui.mask_combobox.setHidden(True)
            self.ui.editMask_btn.setHidden(False)
        else:
            self.ui.mask_combobox.setHidden(True)
            self.ui.editMask_btn.setHidden(False)

        self.ui.usrMan_btn.clicked.connect(self.open_usr_man)
        self.ui.maskType_custom_rdBtn.clicked.connect(self.tgl_cstm)
        self.ui.maskType_pre_rdBtn.clicked.connect(self.tgl_pre)
        self.ui.browse_btn.clicked.connect(self.browse_btn)
        self.ui.editMask_btn.clicked.connect(self.edit_mask)
        self.ui.mask_combobox.currentTextChanged.connect(self.mask_combobox_changed)
        self.ui.complete_btn.clicked.connect(self.complete_btn)
        self.ui.quality_good_rdBtn.toggled.connect(self.good_quality_rdBtn)
        self.ui.quality_better_rdBtn.toggled.connect(self.better_quality_rdBtn)
        self.ui.quality_best_rdBtn.toggled.connect(self.best_quality_rdBtn)

        self.image_size = 64
        self.image_shape = [64, 64, 3]
        self.centerScale = 0.25
        self.nIter = 1000
        self.maskType = 'custom'
        self.inputImgPath = ""
        self.inputImgName = ""
        self.openfaceImgPath = ""
        self.customMaskPath = ""
        self.outImg_path = ""
        self.img_filters = "Image files (*.jpg *.jpeg *.png)"
        self.input_path = os.path.join(os.path.dirname(os.getcwd()), 'input')
        self.mask_path = os.path.join(os.path.dirname(os.getcwd()), 'mask')
        self.maskPrev_path = os.path.join(os.path.dirname(os.getcwd()), 'mask_preview')
        self.output_path = os.path.join(os.path.dirname(os.getcwd()), 'output')
        self.openface_path = os.path.join(os.path.dirname(os.getcwd()), 'openface_out')
        self.usr_man_path = os.path.join(os.path.dirname(os.getcwd()), 'UserMan.pdf')

        self.clean_folder(self.input_path)
        self.clean_folder(self.openface_path)
        self.clean_folder(self.maskPrev_path)
        self.clean_folder(self.mask_path)

    def enable_all_btns(self, bool_flag):
        self.ui.browse_btn.setEnabled(bool_flag)
        self.ui.usrMan_btn.setEnabled(bool_flag)
        self.ui.mask_combobox.setEnabled(bool_flag)
        self.ui.editMask_btn.setEnabled(bool_flag)
        self.ui.complete_btn.setEnabled(bool_flag)
        self.ui.maskType_pre_rdBtn.setEnabled(bool_flag)
        self.ui.maskType_custom_rdBtn.setEnabled(bool_flag)
        self.ui.quality_good_rdBtn.setEnabled(bool_flag)
        self.ui.quality_best_rdBtn.setEnabled(bool_flag)
        self.ui.quality_better_rdBtn.setEnabled(bool_flag)

    def play_loading_gif(self):
        self.enable_all_btns(False)
        self.ui.loadgif_lbl.setHidden(False)
        self.movie.start()

    def stop_loading_gif(self):
        self.enable_all_btns(True)
        self.ui.loadgif_lbl.setHidden(True)

    def open_usr_man(self):
        subprocess.call(['open ' + self.usr_man_path], shell=True)

    def complete_btn(self):
        self.play_loading_gif()
        self.run_complete_subproc()
        self.ui.output_lblPixmap.setPixmap(QPixmap(self.outImg_path)
                                           .scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio))
        self.ui.output_lblPixmap.repaint()

    def browse_btn(self):
        fname = QFileDialog.getOpenFileName(None, 'Browse input', '/Users/Joe/Pictures', self.img_filters)
        if fname[0] != "":
            if len(os.listdir(self.maskPrev_path)) > 0:
                self.clean_folder(self.input_path)
                self.clean_folder(self.openface_path)
            self.inputImgPath = fname[0]
            self.inputImgName = os.path.basename(self.inputImgPath).split('.')[0]
            self.play_loading_gif()
            self.run_openface_subproc()
            self.stop_loading_gif()
            self.openfaceImgPath = os.path.join(self.openface_path, self.inputImgName) + '.png'
            self.ui.input_lblPixmap.setPixmap(QPixmap(self.openfaceImgPath)
                                              .scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio))
            self.init_prev_mask()
            self.ui.output_lblPixmap.clear()

    def mask_combobox_changed(self):
        k = self.edit_mask_warning()
        if k == QMessageBox.Cancel:
            return
        self.ui.output_lblPixmap.clear()
        self.clean_folder(self.maskPrev_path)
        self.clean_folder(self.mask_path)
        self.maskType = self.ui.mask_combobox.currentText().lower()
        prev_img = self.get_prev_mask(np.array(Image.open(self.openfaceImgPath), dtype=np.uint8))
        im = Image.fromarray(prev_img).convert("RGBA")
        pixdata = im.load()
        width, height = im.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)
        im.save(os.path.join(self.maskPrev_path, self.inputImgName + '.png'), "PNG")
        self.ui.mask_lblPixmap.setPixmap(QPixmap(os.path.join(self.maskPrev_path, self.inputImgName + '.png'))
                                         .scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio))

    def init_prev_mask(self):
        subprocess.call('cp ' + self.openfaceImgPath + ' ' + self.maskPrev_path + ' \n', shell=True)
        subprocess.call('cp ' + os.path.join(self.maskPrev_path, self.inputImgName + '.png') +
                        ' ' + self.mask_path + ' \n', shell=True)
        i = Image.open(os.path.join(self.mask_path, self.inputImgName + '.png'))
        img2 = np.array(i)
        mask = np.logical_or(img2[..., 0] > 0, img2[..., 1] > 0, img2[..., 2] > 0)
        img2[mask] = 1
        prev_path = os.path.join(self.maskPrev_path, self.inputImgName + '.png')
        im = Image.open(prev_path).convert("RGBA")
        pixdata = im.load()
        width, height = im.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)
        im.save(prev_path, "PNG")
        self.customMaskPath = os.path.join(self.mask_path, 'currMask.png')
        Image.fromarray(img2).save(self.customMaskPath)
        self.ui.mask_lblPixmap.setPixmap(QPixmap(prev_path)
                                         .scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio))

    def edit_mask(self):
        k = self.edit_mask_warning()
        if k == QMessageBox.Cancel:
            return
        self.ui.output_lblPixmap.clear()
        self.clean_folder(self.maskPrev_path)
        self.clean_folder(self.mask_path)
        m = QMessageBox()
        m.setWindowTitle("On saving your mask")
        m.setIcon(QMessageBox.Information)
        m.setText("Please save your masked image and remember to (QUIT) Preview app and not just close the window")
        m.setDetailedText("Failing to do this will result in app crashes.")
        m.setStandardButtons(QMessageBox.Ok)
        m.exec_()
        subprocess.call('cp ' + self.openfaceImgPath + ' ' + self.maskPrev_path + ' \n', shell=True)
        p = subprocess.Popen(["open", os.path.join(self.maskPrev_path, self.inputImgName + '.png')])
        p.wait()
        plist = [p.name() for p in psutil.process_iter()]
        while "Preview" in plist:
            try:
                plist = [p.name() for p in psutil.process_iter()]
            except psutil.ZombieProcess:
                continue
        subprocess.call('cp ' + os.path.join(self.maskPrev_path, self.inputImgName + '.png') +
                        ' ' + self.mask_path + ' \n', shell=True)
        i = Image.open(os.path.join(self.mask_path, self.inputImgName + '.png'))
        img2 = np.array(i)
        mask = np.logical_or(img2[..., 0] > 0, img2[..., 1] > 0, img2[..., 2] > 0)
        img2[mask] = 1
        self.customMaskPath = os.path.join(self.mask_path, 'currMask.png')
        Image.fromarray(img2).save(self.customMaskPath)
        self.ui.mask_lblPixmap.setPixmap(QPixmap(os.path.join(self.maskPrev_path, self.inputImgName + '.png'))
                                         .scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio))

    def edit_mask_warning(self):
        if len(os.listdir(self.maskPrev_path)) > 0:
            k = QMessageBox()
            k.setWindowTitle("Change mask?")
            k.setText("Are you sure you want to edit your mask?")
            k.setInformativeText("This action cannot be undone")
            k.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            k.setDefaultButton(QMessageBox.Ok)
            k.setIcon(QMessageBox.Warning)
            return k.exec_()

    def get_prev_mask(self, img):
        mask = np.ones(self.image_shape)
        if self.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif self.maskType == 'center':
            assert (self.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size * self.centerScale)
            u = int(self.image_size * (1.0 - self.centerScale))
            mask[l:u, l:u, :] = 0.0
        elif self.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:, :c, :] = 0.0
        elif self.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4, ::4, :] = 1.0
        mask = mask.astype(np.uint8)
        return np.multiply(mask, img, dtype=np.uint8)

    def tgl_cstm(self):
        self.ui.mask_combobox.setHidden(True)
        self.ui.editMask_btn.setHidden(False)
        self.maskType = 'custom'

    def tgl_pre(self):
        self.ui.mask_combobox.setHidden(False)
        self.ui.editMask_btn.setHidden(True)
        self.maskType = self.ui.mask_combobox.currentText()

    def good_quality_rdBtn(self):
        if self.ui.quality_good_rdBtn.isChecked():
            self.nIter = 1000

    def better_quality_rdBtn(self):
        if self.ui.quality_better_rdBtn.isChecked():
            self.nIter = 1000

    def best_quality_rdBtn(self):
        if self.ui.quality_best_rdBtn.isChecked():
            self.nIter = 1200

    def run_complete_subproc(self):
        if self.maskType == 'custom':
            cmd = '/usr/local/bin/python3 /Users/Joe/Desktop/gp/Application_GUI/dcgan_code/complete.py ' \
                  + self.openfaceImgPath + \
                  ' --outDir ' + '/Users/Joe/Desktop/gp/Application_GUI/dcgan_code/output ' \
                  '--maskType custom ' \
                  '--customMask ' + self.customMaskPath \
                  + ' --nIter ' + str(self.nIter + 1)
        else:
            cmd = '/usr/local/bin/python3 /Users/Joe/Desktop/gp/Application_GUI/dcgan_code/complete.py ' \
                  + self.openfaceImgPath + \
                  ' --outDir ' + '/Users/Joe/Desktop/gp/Application_GUI/dcgan_code/output ' \
                  ' --maskType ' + self.maskType \
                  + ' --nIter ' + str(self.nIter + 1)
        subprocess.call([cmd], shell=True)
        self.outImg_path = os.path.join('/Users/Joe/Desktop/gp/Application_GUI/dcgan_code/output/',
                                        'completed_' + self.maskType + '/0_' + str(self.nIter) + '.png')
        subprocess.call(['cp ' + self.outImg_path + ' ' + self.gen_out_img_name()], shell=True)
        self.stop_loading_gif()

    def run_openface_subproc(self):
        subprocess.call('cp ' + self.inputImgPath + ' ' + self.input_path + ' \n'
                        'source ~/virtenv/bin/activate\n'
                        '/Users/Joe/openface/util/align-dlib.py ' + self.input_path +
                        ' align innerEyesAndBottomLip ' + self.openface_path +
                        ' --size 64\n'
                        'cd ' + self.openface_path + '\n'
                        'find . -name \'*.png\' -exec mv {} . \;\n'
                        'find . -type d -empty -delete\n'
                        'cd ~\n'
                        'deactivate\n', shell=True)
        self.stop_loading_gif()

    def clean_folder(self, path):
        subprocess.call('find ' + path + ' -type f -execdir mv \'{}\' ' +
                        os.path.join(os.path.dirname(os.getcwd()), 'trash') + ' \\;', shell=True)

    def gen_out_img_name(self, str_len=8):
        rndd = str(uuid.uuid4())
        rndd = rndd.upper()
        rndd = rndd.replace("-", "")
        return self.output_path + '/' + rndd[0:str_len] + '.png'

    def show(self):
        self.main_win.show()


def viewForm():
    app = QApplication(sys.argv)
    mainWin = mainWindow()
    mainWin.show()
    print('done')
    sys.exit(app.exec_())


if __name__ == "__main__":
    viewForm()
