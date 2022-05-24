#include "mainwindow.h"
#include "./ui_mainwindow.h"

Classifier classifier(0);

MainWindow::MainWindow(QWidget *parent)
        : QMainWindow(parent)
        , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //// Initialize the Background
    QImage image("./image/White.jpg");
    ui->Picture_Box->setPixmap(QPixmap::fromImage(image));
    ui->Picture_Box->setScaledContents(true);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_Training_clicked()
{
    QMessageBox::information(nullptr, tr("Information"), "Training Starting!");

    std::string root = "./dataset";
    QString batch_size, nums_epochs, learning_rate;

    // Default Value
    int epoch=30, batch=16;
    float lr=0.00003;

    // if we input the value
    if(!ui->InputBatch->text().isEmpty()) {
        batch_size = ui->InputBatch->text();
        batch = std::stoi(batch_size.toStdString());
    }
    if(!ui->InputEpoch->text().isEmpty()){
        nums_epochs = ui->InputEpoch->text();
        batch = std::stoi(nums_epochs.toStdString());
    }

    if(!ui->InputLearning->text().isEmpty()){
        learning_rate = ui->InputLearning->text();
        learning_rate = std::stof(learning_rate.toStdString());
    }

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%Y%m%d_%H%M%S",timeinfo);

    std::string time(buffer);
    std::string save_path = "./model/" + time +"_checkpoint_model.pt";

    std::cout << "asdfas" << std::endl;


    classifier.Train(epoch, batch, lr,
                 root, ".jpeg", save_path);
    QMessageBox::information(nullptr, tr("Information"), "Training Finished!");
}


void MainWindow::on_Selected_Image_clicked()
{
    filename = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                            tr("Image Files(*.jpg *.jpeg *.png)"));
    if(!filename.isNull()){
        QImage image(filename);
        if(!image.isNull()){
            ui->Picture_Box->setPixmap(QPixmap::fromImage(image));
            ui->Picture_Box->setScaledContents(true);
            ui->Filename->setText("File: " + filename);
        }
    }else QMessageBox::information(nullptr, tr("ERROR"), "Open Failed");
}

void MainWindow::on_TestSelectedImage_clicked()
{
    // open image
    QImage image(filename);

    // load model
    classifier.LoadWeight(modelname.toStdString());

    // use a hashtable to record the label name
    std::unordered_map<int, QString> label_name;
    label_name[0] = "NORMAL";
    label_name[1] = "PNEUMONIA";

    // log
    std::ofstream writefile("./test_log.txt", std::ios::app);

    // inference
    torch::Tensor prediction;

    if(!image.isNull()){
        auto img = cv::imread(filename.toStdString());
        int plabel;
        plabel=classifier.Inference(img, prediction);
        ui->Prediction->setText("Predict Classes: " + label_name[plabel]);
//        std::cout << plabel << std::endl;
//        std::cout << prediction << std::endl;

        QString Normal_Percentage = QString::number(prediction[0][0].item<float>());
        ui->Normal_P->setText("NORMAL(%): " + Normal_Percentage);

        QString Pneumonia_Percentage = QString::number(prediction[0][1].item<float>());
        ui->PNEUMONIA_P->setText("PNEUMONIA(%): " + Pneumonia_Percentage);
//        std::cout << prediction[0][0].item<float>() << std::endl;

        // write to the log file
        if(writefile.is_open()){
            writefile << "File: " << filename.toStdString() << " | Prediction: " << label_name[plabel].toStdString() <<
            " | NORMAL Percentage: " << Normal_Percentage.toStdString() <<
            " | PNEUMONIA Percentage: " << Pneumonia_Percentage.toStdString() << std::endl;
        }
    }
    writefile.close();
}
void MainWindow::on_SelectModel_clicked()
{
    modelname = QFileDialog::getOpenFileName(this, tr("Open Model Weight"), ".",
                                            tr("Model Files(*.pt)"));
    if(!modelname.isNull()){
        classifier.LoadWeight(modelname.toStdString());
    }else QMessageBox::information(nullptr, tr("ERROR"), "Open Failed");
}