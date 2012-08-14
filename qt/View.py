from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

class MyMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.errorHandler = QtGui.QErrorMessage()
        self.modifyGui()

    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        """
            you should do a fit
        """
        print "crap"
        
    @QtCore.Slot(unicode)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        print arg_1
       
    @QtCore.Slot(unicode)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        """
        model selection changed, update view with parameters from model.
        """
        print arg_1
         
    @QtCore.Slot(int)
    def on_use_errors_checkbox_stateChanged(self, arg_1):
        """
        want to weight by error bars, recalculate chi2
        """
        print arg_1
    
    @QtCore.Slot(QtGui.QTableWidgetItem)
    def on_baseparams_tableWidget_itemChanged(self, arg_1):
        """
       
        """
        row = self.ui.baseparams_tableWidget.currentRow()
        col = self.ui.baseparams_tableWidget.currentColumn()
        
        if row == 0 and col == 0:
            #increase/decrease number of layers
            validator = QtGui.QIntValidator()
            voutput = validator.validate(arg_1.text(), 1)
                        
            if voutput[0] is QtGui.QValidator.State.Acceptable and voutput[1] >= 0:
                #update model
                pass
            else:
                self.errorHandler.showMessage("Number of layers must be integer > 0")
        else:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(arg_1.text(), 1)
            if voutput[0] is QtGui.QValidator.State.Acceptable:
                #update model
                pass
            else:
                self.errorHandler.showMessage("values entered must be numeric")
        
    @QtCore.Slot(QtGui.QTableWidgetItem)
    def on_layerparams_tableWidget_itemChanged(self, arg_1):
        """
       
        """
        row = self.ui.layerparams_tableWidget.currentRow()
        col = self.ui.layerparams_tableWidget.currentColumn()
        numrows = self.ui.layerparams_tableWidget.rowCount()
        numcols = self.ui.layerparams_tableWidget.columnCount()
        if row < 0 or col < 0:
            return
            
        if (row == 0 and col == 0) or (row == numrows - 1 and col == 0) or (row == 0 and col == numcols - 1):
            arg_1.setText("")
            return
            
        validator = QtGui.QDoubleValidator()
        if validator.validate(arg_1.text(), 1)[0] == QtGui.QValidator.State.Acceptable:
            #update model
            pass
        else:
            self.errorHandler.showMessage("values entered must be numeric")

    def modifyGui(self):
        self.ui.baseparams_tableWidget.setHorizontalHeaderLabels(['number of layers', 'scale', 'background'])
        header = self.ui.baseparams_tableWidget.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

        initvals = ["1", "1.0", "1e-7"]
        for cidx in xrange(3):
            wi = QtGui.QTableWidgetItem(initvals[cidx])
            if cidx:
                wi.setCheckState(QtCore.Qt.Unchecked)
            self.ui.baseparams_tableWidget.setItem(0, cidx, wi)

        numrows = self.ui.layerparams_tableWidget.rowCount()
        numcols = self.ui.layerparams_tableWidget.columnCount()
        self.ui.layerparams_tableWidget.setHorizontalHeaderLabels(['thickness', 'sld', 'iSLD', 'roughness'])
        self.ui.layerparams_tableWidget.setVerticalHeaderLabels(['fronting', '1', 'backing'])
        
        initvals = [" ", "0", "0", " ", "25", "3.47", "0", "3", " ", "2.07", "0", "3"]
        idx = 0
        for ridx in xrange(self.ui.layerparams_tableWidget.rowCount()):
            for cidx in xrange(self.ui.layerparams_tableWidget.columnCount()):
                wi = QtGui.QTableWidgetItem(initvals[idx])
                if not ((ridx, cidx) == (0,0) or (ridx, cidx) == (0, 3) or (ridx, cidx) == (numrows - 1, 0)):
                    wi.setCheckState(QtCore.Qt.Unchecked)
                self.ui.layerparams_tableWidget.setItem(ridx, cidx, wi)
                idx += 1
                
        header = self.ui.layerparams_tableWidget.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        header = self.ui.layerparams_tableWidget.verticalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        self.ui.dataset_comboBox.addItem("theoretical")
        