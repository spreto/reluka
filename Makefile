#------------------------------------------------------------------------------#
# This makefile was generated by 'cbp2make' tool rev.1                         #
#------------------------------------------------------------------------------#


WORKDIR = $PWD

CC = gcc
CXX = g++
AR = ar
LD = g++
WINDRES = windres

INC = 
CFLAGS = -Wall -std=c++17 -fexceptions
RCFLAGS = 
RESINC = 
LIBDIR = 
LIB = -lprotobuf -lsoplex -lpthread -lz
LDFLAGS = 

INC_RELEASE = $(INC) -Iinclude -Iinclude/pwl2limodsat -Iinclude/onnx
CFLAGS_RELEASE = $(CFLAGS) -O2
RESINC_RELEASE = $(RESINC)
RCFLAGS_RELEASE = $(RCFLAGS)
LIBDIR_RELEASE = $(LIBDIR)
LIB_RELEASE = $(LIB)
LDFLAGS_RELEASE = $(LDFLAGS) -s
OBJDIR_RELEASE = obj/Release
DEP_RELEASE = 
OUT_RELEASE = bin/Release/reluka

OBJ_RELEASE = $(OBJDIR_RELEASE)/src/pwl2limodsat/VariableManager.o $(OBJDIR_RELEASE)/src/pwl2limodsat/RegionalLinearPiece.o $(OBJDIR_RELEASE)/src/pwl2limodsat/PiecewiseLinearFunction.o $(OBJDIR_RELEASE)/src/pwl2limodsat/LinearPiece.o $(OBJDIR_RELEASE)/src/pwl2limodsat/Formula.o $(OBJDIR_RELEASE)/src/onnx/onnx-ml.proto3.pb.o $(OBJDIR_RELEASE)/src/VnnlibProperty.o $(OBJDIR_RELEASE)/src/OnnxParser4ACASXu.o $(OBJDIR_RELEASE)/src/OnnxParser.o $(OBJDIR_RELEASE)/src/NeuralNetwork.o $(OBJDIR_RELEASE)/src/InequalityConstraints.o $(OBJDIR_RELEASE)/src/GlobalRobustness.o $(OBJDIR_RELEASE)/main.o

all: release

clean: clean_release

before_release: 
	test -d bin/Release || mkdir -p bin/Release
	test -d $(OBJDIR_RELEASE)/src/pwl2limodsat || mkdir -p $(OBJDIR_RELEASE)/src/pwl2limodsat
	test -d $(OBJDIR_RELEASE)/src/onnx || mkdir -p $(OBJDIR_RELEASE)/src/onnx
	test -d $(OBJDIR_RELEASE)/src || mkdir -p $(OBJDIR_RELEASE)/src
	test -d $(OBJDIR_RELEASE) || mkdir -p $(OBJDIR_RELEASE)

after_release: 

release: before_release out_release after_release

out_release: before_release $(OBJ_RELEASE) $(DEP_RELEASE)
	$(LD) $(LIBDIR_RELEASE) -o $(OUT_RELEASE) $(OBJ_RELEASE)  $(LDFLAGS_RELEASE) $(LIB_RELEASE)

$(OBJDIR_RELEASE)/src/pwl2limodsat/VariableManager.o: src/pwl2limodsat/VariableManager.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/pwl2limodsat/VariableManager.cpp -o $(OBJDIR_RELEASE)/src/pwl2limodsat/VariableManager.o

$(OBJDIR_RELEASE)/src/pwl2limodsat/RegionalLinearPiece.o: src/pwl2limodsat/RegionalLinearPiece.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/pwl2limodsat/RegionalLinearPiece.cpp -o $(OBJDIR_RELEASE)/src/pwl2limodsat/RegionalLinearPiece.o

$(OBJDIR_RELEASE)/src/pwl2limodsat/PiecewiseLinearFunction.o: src/pwl2limodsat/PiecewiseLinearFunction.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/pwl2limodsat/PiecewiseLinearFunction.cpp -o $(OBJDIR_RELEASE)/src/pwl2limodsat/PiecewiseLinearFunction.o

$(OBJDIR_RELEASE)/src/pwl2limodsat/LinearPiece.o: src/pwl2limodsat/LinearPiece.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/pwl2limodsat/LinearPiece.cpp -o $(OBJDIR_RELEASE)/src/pwl2limodsat/LinearPiece.o

$(OBJDIR_RELEASE)/src/pwl2limodsat/Formula.o: src/pwl2limodsat/Formula.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/pwl2limodsat/Formula.cpp -o $(OBJDIR_RELEASE)/src/pwl2limodsat/Formula.o

$(OBJDIR_RELEASE)/src/onnx/onnx-ml.proto3.pb.o: src/onnx/onnx-ml.proto3.pb.cc
	$(CC) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/onnx/onnx-ml.proto3.pb.cc -o $(OBJDIR_RELEASE)/src/onnx/onnx-ml.proto3.pb.o

$(OBJDIR_RELEASE)/src/VnnlibProperty.o: src/VnnlibProperty.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/VnnlibProperty.cpp -o $(OBJDIR_RELEASE)/src/VnnlibProperty.o

$(OBJDIR_RELEASE)/src/OnnxParser4ACASXu.o: src/OnnxParser4ACASXu.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/OnnxParser4ACASXu.cpp -o $(OBJDIR_RELEASE)/src/OnnxParser4ACASXu.o

$(OBJDIR_RELEASE)/src/OnnxParser.o: src/OnnxParser.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/OnnxParser.cpp -o $(OBJDIR_RELEASE)/src/OnnxParser.o

$(OBJDIR_RELEASE)/src/NeuralNetwork.o: src/NeuralNetwork.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/NeuralNetwork.cpp -o $(OBJDIR_RELEASE)/src/NeuralNetwork.o

$(OBJDIR_RELEASE)/src/InequalityConstraints.o: src/InequalityConstraints.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/InequalityConstraints.cpp -o $(OBJDIR_RELEASE)/src/InequalityConstraints.o

$(OBJDIR_RELEASE)/src/GlobalRobustness.o: src/GlobalRobustness.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/GlobalRobustness.cpp -o $(OBJDIR_RELEASE)/src/GlobalRobustness.o

$(OBJDIR_RELEASE)/main.o: main.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c main.cpp -o $(OBJDIR_RELEASE)/main.o

clean_release: 
	rm -f $(OBJ_RELEASE) $(OUT_RELEASE)
	rm -rf bin/Release
	rm -rf $(OBJDIR_RELEASE)/src/pwl2limodsat
	rm -rf $(OBJDIR_RELEASE)/src/onnx
	rm -rf $(OBJDIR_RELEASE)/src
	rm -rf $(OBJDIR_RELEASE)

.PHONY: before_release after_release clean_release

