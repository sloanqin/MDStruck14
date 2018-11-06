function [ feat_conv_out ] = prep_feat_conv_data( feat_conv_in )
% prep_feat_conv_data
% prepare data for feat_conv
%
% INPUT:
%   x_ind  - frame index to prepare x data
%
% OUTPUT:
%   svs_feats - features of support vectors
%   svs_betas - beta of support vectors
%   kernerl_sigma - sigma for kernerl function
%
% Sloan Qin, 2017
% 

feat_conv_out = max(max(feat_conv_in));

% qyy normlization
x_max = max(max(feat_conv_out));
x_min = min(min(feat_conv_out));
feat_conv_out = (feat_conv_out-x_min)/(x_max-x_min);

end
