Introduction of Gastric Histopathology Subsize Image Database (GasHisSDB)

GasHisSDB is a New Gastric Histopathology Subsize Image Database with a 
total of 245196 images.
GasHisSDB is divided into 160x160 pixels sub-database, 120x120 pixels 
sub-database and 80x80 pixels sub-database.
Each sub-database has two types of images: normal and abnormal.

We state the problem of data splitting as follows： The ratio of the training, validation and test sets is split 4:4:2  in our experiments. For the traditional machine learning experiment part, since we randomly shuffle the order of the images when creating the data set, we use the first 40% as the training set and the next 20% as the test set. This part does not involve the validation set, but we have to choose this way to ensure the accuracy of the comparative experiment. For the deep learning experiment part, we randomly select 80% of the images, and randomly divide them into two equal parts as the training set and the validation set, and the remaining 20% is the test set.


If you need it or have any questions, please contact us.

First release: 10-06-2021.

Nearst update: 28-07-2021.

Any questions: Prof. Dr.-Ing. Chen Li, lichen201096@hotmail.com, lichen@bmie.neu.edu.cn

Related people: Chen Li, Weiming Hu, Changhao Sun