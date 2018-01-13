#ifndef _N_HPP
#define _N_HPP

#include <string>

enum ActivationFunction
{
	linear, ReLU, PReLU, ELU, Tanh, Sigmoid
};

void setActiv(ActivationFunction);
void endN();

class Neuron
{
public:

	double bias;				//b
	double value;				//x
	int nextLayerCardinality;	//Licznoœæ nastêpnej warstwy
	double gradient;
	double* weights;			//w
	std::string type;			//Input, Hidden, Output
	double* m;					//Parametr potrzebny do algorytmu Adam
	double* v;					//Parametr potrzebny do algorytmu Adam

	Neuron(int nLC, double* w, std::string t);	//Tworzy neuron o Licznosci nastêpnej warstwy = "nLC", wagach "w" i typie "t"
	Neuron();									//Domyœlny konstruktow
	void reset();								//Resetuje parametry algorytmu Adam
	void activate();							//Funkcja aktywacji
	void reActivate();							//Cofniêta aktywacja (faza Backpropagation)
	void writeToFile(std::fstream* plik);		//Zapisanie do pliku
	void readFromFile(std::fstream* plik);		//Wczytanie z pliku
};

#endif