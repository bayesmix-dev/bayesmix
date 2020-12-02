LOSS_DIR = lossfunction
OBJ_FILES_LOSS = $(LOSS_DIR)/LossFunction.o $(LOSS_DIR)/VariationInformation.o $(LOSS_DIR)/BinderLoss.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	g++  -Wall -O -c $<

compileLoss: $(OBJ_FILES_LOSS)
	g++ -lm -o loss $(LOSS_DIR)/loss.cpp $(LOSS_DIR)/BinderLoss.o $(LOSS_DIR)/VariationInformation.o $(LOSS_DIR)/LossFunction.o

loss: compileLoss
	./$(LOSS_DIR)/loss

clean:
	rm -f $(LOSS_DIR)/*.o