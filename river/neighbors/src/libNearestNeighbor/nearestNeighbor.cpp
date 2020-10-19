#include "Python.h"
#include "numpy/arrayobject.h"
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <limits>
#include <map>


using namespace std;

void nArgMin(const int &n, const double* values, int* indices, const int &numValues){
	for (int i=0; i<n; i++){
		double minValue = std::numeric_limits<double>::max();
		for (int j=0; j<numValues; j++){
			if (values[j] < minValue){
				bool alreadyUsed = false;
				for (int k=0; k<i; k++){
					if (indices[k]==j){
						alreadyUsed = true;
					}
				}
				if (!alreadyUsed){
					indices[i] = j;
					minValue = values[j];
				}
			}
		}
	}
}
void nArgMinN(const int &n, const double* values, int* indices, const int &nRows, const int &numValues){
	for (int i=0; i<nRows; i++)
	{
		nArgMin(n, &values[i*numValues], &indices[i*n], numValues);
	}
}

void mostCommon(const int* values, int* result, const int &numValues){
	map<int,int> dict;
	for (int i=0; i<numValues; i++)
	{
		if (dict.find(values[i])==dict.end()){
			dict[values[i]] = 1;
		} else{
			dict[values[i]]++;
		}
	}
	int maxValue = 0;
	int maxKey = -1;
	for(std::map<int,int>::iterator iter = dict.begin(); iter != dict.end(); ++iter)
	{
		 if (iter->second > maxValue){
			 maxValue = iter->second;
			 maxKey = iter->first;
		 }
	}
	*result = maxKey;
}


void mostCommonN(const int* values, int* result, const int &nRows, const int &numValues){
	for (int i=0; i<nRows; i++)
	{
		mostCommon(&values[i*numValues], &result[i], numValues);
	}
}

void linearWeightedLabels(const int* labels, const double* distances, int* result, const int &numValues){
	map<int,double> dict;
	for (int i=0; i<numValues; i++)
	{
		if (dict.find(labels[i])==dict.end()){
			dict[labels[i]] = 1/max(distances[i], 0.000000001);
		} else{
			dict[labels[i]] += 1/max(distances[i], 0.000000001);
		}
	}
	double maxValue = 0;
	int maxKey = -1;
	for(std::map<int,double>::iterator iter = dict.begin(); iter != dict.end(); ++iter)
	{
		 if (iter->second > maxValue){
			 maxValue = iter->second;
			 maxKey = iter->first;
		 }
	}
	*result = maxKey;
}

void linearWeightedLabelsN(const int* labels, const double* distances, int* result, const int &nRows, const int &numValues){
	for (int i=0; i<nRows; i++)
	{
		linearWeightedLabels(&labels[i*numValues], &distances[i*numValues], &result[i], numValues);
	}
}

double getDistance(const double* sample, const double *sample2, const int &numFeatures)
{
	double sum=0;
	for (int i=0; i<numFeatures; i++)
	{
		double diff = sample[i]-sample2[i];
		sum += diff*diff;
	}
	return sum;
}

void get1ToNDistances(const double* sample, const double* samples, double* distances, const int& numSamples, const int& numFeatures){
	for (int i=0; i<numSamples; i++){
		distances[i] = getDistance(sample, &samples[i*numFeatures], numFeatures);
	}
}

void getNToNDistances(const double* samples, const double* samples2, double* distances, const int& numSamples, const int& numSamples2, const int& numFeatures){
	for (int i=0; i<numSamples; i++){
		get1ToNDistances(&samples[i*numFeatures], samples2, &distances[numSamples2*i], numSamples2, numFeatures);
	}
}



static PyObject *py_getNToNDistances(PyObject *self, PyObject *args) {
	PyObject *sampleData;
	PyObject *sampleData2;
	if (!PyArg_ParseTuple(args, "OO", &sampleData, &sampleData2))
		return NULL;

	PyArrayObject *matSampleData;
	matSampleData = (PyArrayObject *) PyArray_ContiguousFromObject(sampleData,
			PyArray_DOUBLE, 1, 2);
	double *carrSampleData = (double*) (matSampleData->data);
	npy_intp sampleDataRows = matSampleData->dimensions[0];
	npy_intp sampleDataCols = matSampleData->dimensions[1];

	if (matSampleData->nd == 1) {
		sampleDataCols = sampleDataRows;
		sampleDataRows = 1;
	}
	PyArrayObject *matSampleData2;
	matSampleData2 = (PyArrayObject *) PyArray_ContiguousFromObject(sampleData2,
			PyArray_DOUBLE, 1, 2);
	double *carrSampleData2 = (double*) (matSampleData2->data);
	npy_intp sampleData2Rows = matSampleData2->dimensions[0];

	if (matSampleData2->nd == 1) {
		sampleData2Rows = 1;
	}
	npy_intp Dims[2];
	Dims[0]= sampleDataRows;
	Dims[1] = sampleData2Rows;

	PyArrayObject *distances = (PyArrayObject *) PyArray_SimpleNew(2,
			Dims, PyArray_DOUBLE);
	PyArrayObject *matDistances;
	matDistances = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )distances, PyArray_DOUBLE, 1, 2);

	double *carrDistances = (double*) (matDistances->data);

	getNToNDistances(carrSampleData, carrSampleData2, carrDistances, sampleDataRows, sampleData2Rows, sampleDataCols);

	Py_DECREF(matSampleData);
	Py_DECREF(matSampleData2);
	PyObject *result = Py_BuildValue("O", matDistances);
	Py_DECREF(matDistances);
	Py_DECREF(distances);
	return result;
}

PyDoc_STRVAR(py_getNToNDistances__doc__,
		"name, data 2D-arr. Returns label result matrix");




static PyObject *py_get1ToNDistances(PyObject *self, PyObject *args) {
	PyObject *sampleData;
	PyObject *samplesData;
	if (!PyArg_ParseTuple(args, "OO", &sampleData, &samplesData))
		return NULL;

	PyArrayObject *matSampleData;
	matSampleData = (PyArrayObject *) PyArray_ContiguousFromObject(sampleData,
			PyArray_DOUBLE, 1, 1);
	double *carrSampleData = (double*) (matSampleData->data);
	npy_intp sampleDataRows = matSampleData->dimensions[0];
	npy_intp sampleDataCols = matSampleData->dimensions[1];

	if (matSampleData->nd == 1) {
		sampleDataCols = sampleDataRows;
		sampleDataRows = 1;
	}

	PyArrayObject *matSamplesData;
	matSamplesData = (PyArrayObject *) PyArray_ContiguousFromObject(samplesData,
			PyArray_DOUBLE, 1, 2);
	double *carrSamplesData = (double*) (matSamplesData->data);
	npy_intp samplesDataRows = matSamplesData->dimensions[0];

	if (matSamplesData->nd == 1) {
		samplesDataRows = 1;
	}

	PyArrayObject *distances = (PyArrayObject *) PyArray_SimpleNew(1,
			&samplesDataRows, PyArray_DOUBLE);
	PyArrayObject *matDistances;
	matDistances = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )distances, PyArray_DOUBLE, 1, 1);
	double *carrDistances = (double*) (matDistances->data);
	get1ToNDistances(carrSampleData, carrSamplesData, carrDistances, samplesDataRows, sampleDataCols);
	Py_DECREF(matSampleData);
	Py_DECREF(matSamplesData);
	PyObject *result = Py_BuildValue("O", matDistances);
	Py_DECREF(matDistances);
	Py_DECREF(distances);
	return result;
}

PyDoc_STRVAR(py_get1ToNDistances__doc__,
		"name, data 2D-arr. Returns label result matrix");

static PyObject *py_nArgMin(PyObject *self, PyObject *args) {
	PyObject *values;
	int n;
	if (!PyArg_ParseTuple(args, "iO", &n, &values))
		return NULL;

	//create c-arrays:
	PyArrayObject *matValues;
	matValues = (PyArrayObject *) PyArray_ContiguousFromObject(values,
			PyArray_DOUBLE, 1, 2);
	double *carrValues = (double*) (matValues->data);
	npy_intp numRows = matValues->dimensions[0];
	npy_intp numValues = matValues->dimensions[1];

	if (matValues->nd == 1) {
		numValues = numRows;
		numRows = 1;
	}
	npy_intp Dims[2];
	Dims[0]= numRows;
	Dims[1] = n;

	PyArrayObject *indices = (PyArrayObject *) PyArray_SimpleNew(2,
			Dims, PyArray_INT);
	PyArrayObject *matIndices;
	matIndices = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )indices, PyArray_INT, 1, 2);
	int *carrIndices = (int*) (matIndices->data);
	nArgMinN(n, carrValues, carrIndices, numRows, numValues);
	Py_DECREF(matValues);
	PyObject *result = Py_BuildValue("O", matIndices);
	Py_DECREF(matIndices);
	Py_DECREF(indices);
	return result;
}

PyDoc_STRVAR(py_nArgMin__doc__,
		"name, data 2D-arr. Returns label result matrix");

static PyObject *py_mostCommon(PyObject *self, PyObject *args) {
	PyObject *values;
	if (!PyArg_ParseTuple(args, "O", &values))
		return NULL;

	//create c-arrays:
	PyArrayObject *matValues;
	matValues = (PyArrayObject *) PyArray_ContiguousFromObject(values,
			PyArray_INT, 1, 2);
	int *carrValues = (int*) (matValues->data);
	npy_intp numRows = matValues->dimensions[0];
	npy_intp numValues = matValues->dimensions[1];

	if (matValues->nd == 1) {
		numValues = numRows;
		numRows = 1;
	}
	PyArrayObject *indices = (PyArrayObject *) PyArray_SimpleNew(1,
			&numRows, PyArray_INT);
	PyArrayObject *matIndices;
	matIndices = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )indices, PyArray_INT, 1, 1);
	int *carrIndices = (int*) (matIndices->data);
	mostCommonN(carrValues, carrIndices, numRows, numValues);
	Py_DECREF(matValues);
	PyObject *result = Py_BuildValue("O", matIndices);
	Py_DECREF(matIndices);
	Py_DECREF(indices);
	return result;
}
PyDoc_STRVAR(py_mostCommon__doc__,
		"name, data 2D-arr. Returns label result matrix");

static PyObject *py_getLinearWeightedLabels(PyObject *self, PyObject *args) {
	PyObject *labels, *distances;
	if (!PyArg_ParseTuple(args, "OO", &labels, &distances))
		return NULL;

	//create c-arrays:
	PyArrayObject *matLabels;
	matLabels = (PyArrayObject *) PyArray_ContiguousFromObject(labels,
			PyArray_INT, 1, 2);
	int *carrLabels = (int*) (matLabels->data);
	npy_intp numRows = matLabels->dimensions[0];
	npy_intp numValues = matLabels->dimensions[1];

	if (matLabels->nd == 1) {
		numValues = numRows;
		numRows = 1;
	}
	PyArrayObject *matDistances;
	matDistances = (PyArrayObject *) PyArray_ContiguousFromObject(distances,
			PyArray_DOUBLE, 1, 2);
	double *carrDistances = (double*) (matDistances->data);

	PyArrayObject *resultLabels = (PyArrayObject *) PyArray_SimpleNew(1,
			&numRows, PyArray_INT);
	PyArrayObject *matResultLabels;
	matResultLabels = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )resultLabels, PyArray_INT, 1, 1);
	int *carrResultLabels = (int*) (matResultLabels->data);
	linearWeightedLabelsN(carrLabels, carrDistances, carrResultLabels, numRows, numValues);
	Py_DECREF(matLabels);
	Py_DECREF(matDistances);
	PyObject *result = Py_BuildValue("O", matResultLabels);
	Py_DECREF(matResultLabels);
	Py_DECREF(resultLabels);
	return result;
}
PyDoc_STRVAR(py_getLinearWeightedLabels__doc__,
		"name, data 2D-arr. Returns label result matrix");

/* The module doc string */
PyDoc_STRVAR(ORF__doc__, "Nearest neighbor python interface");



/* A list of all the methods defined by this module. */
/* "iterate_point" is the name seen inside of Python */
/* "py_iterate_point" is the name of the C function handling the Python call */
/* "METH_VARGS" tells Python how to call the handler */
/* The {NULL, NULL} entry indicates the end of the method definitions */
static PyMethodDef NN_methods[] = { {"getNToNDistances", py_getNToNDistances, METH_VARARGS, py_getNToNDistances__doc__},
		{"get1ToNDistances", py_get1ToNDistances, METH_VARARGS, py_get1ToNDistances__doc__},
		{"nArgMin", py_nArgMin, METH_VARARGS, py_nArgMin__doc__},
		{"mostCommon", py_mostCommon, METH_VARARGS, py_mostCommon__doc__},
		{"getLinearWeightedLabels", py_getLinearWeightedLabels, METH_VARARGS, py_getLinearWeightedLabels__doc__},
		{ NULL, NULL } /* sentinel */
};

/* When Python imports a C module named 'X' it loads the module */
/* then looks for a method named "init"+X and calls it.  Hence */
/* for the module "mandelbrot" the initialization function is */
/* "initmandelbrot".  The PyMODINIT_FUNC helps with portability */
/* across operating systems and between C and C++ compilers */
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_libNearestNeighbor(void)
#else
initlibNearestNeighbor(void)
#endif
{
	import_array()

	#if PY_MAJOR_VERSION >= 3
		static struct PyModuleDef moduledef = {
			PyModuleDef_HEAD_INIT,
			"libNearestNeighbor",
			ORF__doc__,
			-1,
			NN_methods,
			NULL,
			NULL,
			NULL,
			NULL,
		};
	#endif

	#if PY_MAJOR_VERSION >= 3
		return PyModule_Create(&moduledef);

	#else
		/* There have been several InitModule functions over time */
		return Py_InitModule3("libNearestNeighbor", NN_methods, ORF__doc__);

	#endif

}


