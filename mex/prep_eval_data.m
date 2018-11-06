function [ svs_feats, svs_beta, kernerl_sigma, xs_feats ] = prep_eval_data( x_ind )
% prep_data_eval
% prepare data fot struct svm eval function, mex c funciton
%
% INPUT:
%   x_ind  - frame index to prepare x data
%
% OUTPUT:
%   svs_feats - features of support vectors
%   svs_betas - beta of support vectors
%   kernerl_sigma - sigma for kernerl function
%   xs_feats - features of x_ind frame's examples
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
global total_data;

[ svs_feats, svs_beta, kernerl_sigma ] = prep_evaluate_data();

xs_feats = squeeze(total_data{1,1,1,x_ind});
xs_feats = double(xs_feats);

end

