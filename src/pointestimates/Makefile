CXXFLAGS = -W -Wall -g

LOSS_DIR = lossfunction
OBJ_FILES_LOSS = $(LOSS_DIR)/LossFunction.o $(LOSS_DIR)/VariationInformation.o $(LOSS_DIR)/BinderLoss.o

CLUSTER_DIR = clustering
OBJ_FILES_CLUSTERING = $(CLUSTER_DIR)/ClusteringProcess.o

$(CLUSTER_DIR)/%.o: $(CLUSTER_DIR)/%.c
	g++ -O -c $< $(CXXFLAGS)

compileLoss: $(OBJ_FILES_LOSS)
	g++ -lm -o ./$(LOSS_DIR)/loss $(LOSS_DIR)/loss.cpp $(LOSS_DIR)/BinderLoss.o $(LOSS_DIR)/VariationInformation.o $(LOSS_DIR)/LossFunction.o $(CXXFLAGS)

compileClustering: $(OBJ_FILES_CLUSTERING) $(OBJ_FILES_LOSS)
	g++ -lm -o pe_estimate point_estimate_test.cpp $(LOSS_DIR)/BinderLoss.o $(LOSS_DIR)/VariationInformation.o $(LOSS_DIR)/LossFunction.o $(CLUSTER_DIR)/ClusteringProcess.o $(CXXFLAGS)


loss: compileLoss
	./$(LOSS_DIR)/loss

pe_test: compileClustering
	./pe_estimate

clean:
	rm -f */*.o