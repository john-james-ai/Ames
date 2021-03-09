make_files <- function(set_id) {

    ames_file = file.path("./data/external/Ames_data.csv")
    test_id_file =  file.path("./data/external/project1_testIDs.dat")

    train_file = file.path("./data/raw/", paste0(set_id,"_train.csv"))
    test_file = file.path("./data/raw/", paste0(set_id,"_test.csv"))
    test_y_file = file.path("./data/raw/", paste0(set_id,"_test_y.csv"))

    data = read.csv(ames_file)
    test_ids = read.table(test_id_file)    
    train = data[-test_ids[,set_id],]
    test = data[test_ids[,set_id],]
    test.y = test[, c(1,83)]
    test = test[,-83]
    write.csv(train, train_file, row.names=FALSE)
    write.csv(test, test_file, row.names=FALSE)
    write.csv(test.y, test_y_file, row.names=FALSE)
}

for (i in 1:10) {
    make_files(i)
    list.files("./data/raw/")
}