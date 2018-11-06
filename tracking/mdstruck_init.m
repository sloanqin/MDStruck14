function mdstruck_init()
% MDSTRUCK_INIT
% Initialize structured svm.
%
% sloan qin, 2017
% 

global st_svm;

% for support patterns
st_svm.supportPatterns = cell(0,1);

% for support vectors
st_svm.supportVectors = cell(0,1);

% save targetScore
st_svm.targetScores = zeros(0,1);

% max number of support vectors
st_svm.kMaxSVs = 2000;

% svmBudgetSize, limit the number of support vectors
st_svm.svmBudgetSize = 100;

% kernel matrix
if st_svm.svmBudgetSize>0
	st_svm.N = st_svm.svmBudgetSize;
else
	st_svm.N = st_svm.kMaxSVs;
end
st_svm.m_k = zeros(st_svm.N,st_svm.N);

% kernerl kind
st_svm.kernerl = 'GaussianKernel'; % LinearKernel,GaussianKernel etc

% GaussianKernel kernel-params
st_svm.kernerl_m_sigma = 0.2;

% SVM regularization parameter.
st_svm.svmC = 10;

% SVM model update target score threshold.
st_svm.update_target_score_threshold = 0.01;

% first_frame_st_svm target score threshold for bbox regression prediction.
st_svm.bbox_reg_score_threshold = 0.1;

% frame interval to replace the old model.
st_svm.snapshot_interval = 25;

% st_svm eval sample trans_f.
st_svm.trans_f = 0.6;

% st_svm eval sample scale_f.
st_svm.scale_f = 1;

end