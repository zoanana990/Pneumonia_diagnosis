#ifndef FINAL_PROJECT_MAINWINDOW_H
#define FINAL_PROJECT_MAINWINDOW_H
#include <QWidget>
#include <QMainWindow>
#include <QLabel>
#include <QImage>
#include <QFileDialog>
#include <QString>
#include <QPixmap>
#include <QMessageBox>
#include <ctime>
#include <fstream>

#undef slots
#include "dataset.h"
#include "Classifier.h"
#define slots Q_SLOTS

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_Training_clicked();

//    void on_Testing_TS_clicked();

    void on_Selected_Image_clicked();

    void on_TestSelectedImage_clicked();

    void on_SelectModel_clicked();

private:
    Ui::MainWindow *ui;
    QString filename;
    QString modelname="./model/Default.pt";
};
#endif //FINAL_PROJECT_MAINWINDOW_H


