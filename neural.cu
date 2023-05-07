/* -------------------------------------------------------------*
* Простая нейронная сеть, реализованная с помощью CUDA и cuBLAS *
* --------------------------------------------------------------*/

// подключение библиотек
#include <cmath>
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WITH_ACTIVATION true
#define WITHOUT_ACTIVATION true
#define PRINT_RESULT true
#define DONT_PRINT false
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// макросы для проверки ошибок CUDA и CUBLAS 
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)


#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


// функция активации. __global__ функции не могут быть членом класса, так что вынесла
__global__ void nn_Sigmoid(float *arr)
{
	int id = threadIdx.x;
	arr[id] = 1 / (1 + exp(-arr[id]));
}

// проверка ответа
void check(float result){
	float pattern = round(0.5696*1000)/1000;
	result = round(result*1000)/1000;

	if(pattern == result) std::cout << "IT'S RIGHT ANSWER!"<< std::endl;
	else std::cout << "result("<< result << ") != pattern(" << pattern << ")" << std::endl;
}

// класс для линейного слоя и сигмоиды
class NN
{
private:
	cublasHandle_t handle;
	float alpha, beta;
	float *weights, *biases, *output;
	int input_size, output_size;
	bool activation_true;

	// считывание весов из файла
	void read_weights(std::string pathToWeights){
		float *host_array = new float [input_size*output_size], *host_array_row = new float [input_size*output_size];
		try{
			std::ifstream fin(pathToWeights);
			for (int i = 0; i < input_size*output_size; i++) fin >> host_array_row[i];
			fin.close();
		}
		catch(std::exception const& e){
			std::cout << "There was an error: " << e.what() << std::endl;
		}
		for(int i=0;i<input_size;i++){
			for(int j=0;j<output_size;j++){
				host_array[i*output_size+j] = host_array_row[IDX2C(i,j, input_size)];
			}
		}
		CUDA_CHECK(cudaMalloc(&weights, output_size * input_size * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(weights, host_array, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array, host_array_row;
	};

	// считывание добавочных членов из файла
	void read_biases(std::string pathToWeights){
		float *host_array = new float [output_size];
		try{
			std::ifstream fin(pathToWeights);
			for (int i = 0; i < output_size; i++) {
				fin >> host_array[i];
				//std::cout << arr[i] << " ";
			}
			fin.close();
		}
		catch(std::exception const& e){
			std::cout << "There was an error: " << e.what() << std::endl;
		}
		CUDA_CHECK(cudaMalloc(&biases, output_size * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(biases, host_array, output_size*sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array;
	};

public:
	// конструкторы
	NN(){
		input_size = 0;
		output_size = 0;
		alpha = 1.0;
		beta = 1.0;
		activation_true = false;
	};

	NN(std::string pathToWeights, std::string pathToBiases, int inSize, int outSize, bool activation){
		alpha = 1.0;
		beta = 1.0;
		input_size = inSize;
		output_size = outSize;
		read_weights(pathToWeights);
		read_biases(pathToBiases);
		activation_true = activation;

	};

	// сигмоида
	void Sigmoid(float *arr)
	{
		nn_Sigmoid<<<1, output_size>>> (arr);
	};

	// линейный слой
	float* Linear(float* input){
		CUBLAS_CHECK(cublasCreate(&handle));
		CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, output_size, input_size, &alpha, 
             weights,output_size, input, 1, &beta, biases, 1));
		CUBLAS_CHECK(cublasDestroy(handle));
		if(activation_true){
			Sigmoid(biases);
		}
		return biases;
	};

	// деструктор
	~NN(){
		if(weights!=nullptr) cudaFree(weights);
		if(biases!=nullptr) cudaFree(biases);
	};
};

// класс для построения сети из функций NN. По умолчанию поставлена сеть из ТЗ,
// но есть возможность сделать свою.
class Net
{
private:
	float *array;
	int input_size, output_size;
	std::vector<NN> layers;

	// чтение входного массива из файла
	void read_input(std::string pathToWeights){
		float *host_array = new float [input_size];
		try{
			std::ifstream fin(pathToWeights);
			for (int i = 0; i < input_size; i++) fin >> host_array[i];
			fin.close();
		}
		catch(std::exception const& e){
			std::cout << "There was an error: " << e.what() << std::endl;
		}

		CUDA_CHECK(cudaMalloc(&array, input_size * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(array, host_array, input_size*sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array;
	};

	// печать
	void print_result(float* arr){
		float* host_array = new float[output_size];
		CUDA_CHECK(cudaMemcpy(host_array, arr, output_size*sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "Result: " << std::endl;
		for (int i = 0; i < output_size; i++)
		{
			std::cout << host_array[i] << std::endl;
		}
		check(host_array[0]);
		delete[] host_array;
	};

public:
	// конструктор по умолчанию
	Net(){
		input_size = 1024;
		output_size = 1;
	};

	// добавление слоя в сеть
	void push_back_linear(std::string pathToWeights, std::string pathToBiases, int inSize, int outSize, bool activation){
		if(layers.size()==0) input_size=inSize;
		output_size = outSize;
		layers.push_back(NN(pathToWeights, pathToBiases, inSize, outSize, activation));
	};

	// запуск своей сети
	void my_forward(std::string pathToFile, bool print){
		read_input(pathToFile);
		for(auto& layer : this->layers){
        	array = layer.Linear(array);
        	}
		if(print) print_result(array);
	};

	// запуск базовой сети
	void forward(std::string pathToFile, bool print){
		read_input(pathToFile);
		NN layer1("weights1.bin", "biases1.bin", 1024, 256, WITH_ACTIVATION);
		array = layer1.Linear(array);
		NN layer2("weights2.bin", "biases2.bin", 256, 16, WITH_ACTIVATION);
		array = layer2.Linear(array);
		NN layer3("weights3.bin", "biases3.bin", 16, 1, WITH_ACTIVATION);
		array = layer3.Linear(array);
		if(print) print_result(array);
	}

	// деструктор
	~Net(){
		if(array!=nullptr) cudaFree(array);
	};
};

int main()
{
	Net model;
	model.forward("inputs1.bin", PRINT_RESULT);
	return 0;
}
