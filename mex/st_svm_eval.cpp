/**
* @file st_svm_eval.cpp
* @mex interface for struct svm eval(st_svm_eval)
* @author yuanyuan qin
* @date 2017.3
*/

//#include "mexopencv.hpp"
#include "mex.h" 
#include <vector>
#include <cmath>
using namespace std;
//using namespace cv;

double st_svm_kernel_eval(vector<double> &x1, vector<double> &x2, double kernerlSigma)
{
	//res = exp(-m_sigma*squaredNorm(x1-x2));

	if (x1.size() != x2.size())
		mexErrMsgIdAndTxt("st_svm_kernel_eval:error", "vectors' dimension are inconsistent");

	//squaredNorm(x1-x2)
	int size = x1.size();
	double squaredNorm = 0.0;
	for (int i = 0; i < size; i++)
	{
		squaredNorm += (x1[i] - x2[i])*(x1[i] - x2[i]);
	}

	double res = exp(-kernerlSigma * squaredNorm);

	return res;
}


double st_svm_evaluate(const mxArray *pSvsFeatsArray, const mxArray *pSvsBetaArray, double kernerlSigma, vector<double> &xFeats)
{
	double f = 0.0;

	double* pSvsFeats = (double*)mxGetPr(pSvsFeatsArray);
	double* pSvsBeta = (double*)mxGetPr(pSvsBetaArray);

	int svsNum = mxGetM(pSvsBetaArray);
	int featsDimension = mxGetM(pSvsFeatsArray);
	for (int i = 0; i < svsNum; i++)
	{
		vector<double> svxFeats(pSvsFeats + i*featsDimension, pSvsFeats + i*featsDimension + featsDimension);
		f += pSvsBeta[i] * st_svm_kernel_eval(svxFeats, xFeats, kernerlSigma);
	}

	return f;
}


/**
* Main entry called from Matlab
* @param nlhs number of left-hand-side arguments
* @param plhs pointers to mxArrays in the left-hand-side
* @param nrhs number of right-hand-side arguments
* @param prhs pointers to mxArrays in the right-hand-side
* @usage [ results ] = st_svm_eval(svs_feats, svs_beta, kernerl_sigma, xs_feats)
*/
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	// Check the number of arguments
	if (nrhs != 4)
		mexErrMsgIdAndTxt("st_svm_eval:error", "Wrong number of arguments input");

	if (nlhs != 1)
		mexErrMsgIdAndTxt("st_svm_eval:error", "Wrong number of arguments output");

	double* pKernerlSigma = (double*)mxGetPr(prhs[2]);
	double* pXsFeats = (double*)mxGetPr(prhs[3]);
	double kernerlSigma = pKernerlSigma[0];

	//mexPrintf("size of pXsFeats is %d,%d  kernerlSigma is %f \n", mxGetM(prhs[3]), mxGetN(prhs[3]), kernerlSigma);

	//output
	int resultNum = mxGetN(prhs[3]);
	plhs[0] = mxCreateDoubleMatrix(1, resultNum, mxREAL);
	double* pResults = (double*)mxGetPr(plhs[0]);

	int featsDimension = mxGetM(prhs[3]);

	//mexPrintf("xs_feats is %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n", );
	for (int i = 0; i < resultNum; i++)
	{
		vector<double> xsFeats(pXsFeats + i*featsDimension, pXsFeats + i*featsDimension + featsDimension);
		//if (i<10)
		//{
		//	mexPrintf("xsFeats begin:%f, end:%f \n", xsFeats[0], xsFeats[featsDimension - 1]);
		//}
		//if (i == (resultNum - 1))
		//{
		//	mexPrintf("xsFeats begin:%f, end:%f \n", xsFeats[0], xsFeats[featsDimension - 1]);
		//}
		pResults[i] = st_svm_evaluate(prhs[0], prhs[1], kernerlSigma, xsFeats);
		//mexPrintf("pResults %d is %f\n", i, pResults[i]);
	}

}



