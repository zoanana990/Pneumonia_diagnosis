/******************************************************************************************
 * author @Hsin Yu Chen
 * Date @2021.12.22
 * Dataset Source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 ******************************************************************************************/
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}