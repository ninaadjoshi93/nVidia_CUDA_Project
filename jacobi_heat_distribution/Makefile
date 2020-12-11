CC = nvcc
SOURCE = heat_distribution_cuda.cu
OBJ = heat_distribution_cuda.o
TARGET = heat_distribution_cuda
LIBS = -lm -lrt
# set the debug and display utilities
DISPLAY = 0
DEBUG = 0
# set the row and column size of the heat plate
COLS = 100
ROWS = 100
# set the block size in the X and Y dimensions
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 32

# Use the following flags to enable debug outputs or display the output
# and set the rows and columns of the heat plate
CFLAGS = -DDEBUG=$(DEBUG) -DDISPLAY=$(DISPLAY) \
		-DROWS=$(ROWS) -DCOLS=$(COLS) \
		-DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y)

target: $(SOURCE)
	@$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
run: $(SOURCE)
	@$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
	@aprun $(TARGET)

clean: $(TARGET)
	@rm -rf $(TARGET)
