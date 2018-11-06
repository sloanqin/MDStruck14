function calcu_similarity(model_scores)
% plot_scores_map
% plot 3D map of examples' scores
%
% INPUT:
%   examples  - parameters of structured svm: support patterns and support vectors
%   scores  - fc4 features and example boxes
%   figure_ind  - index of figure to plot
%
% Sloan Qin, 2017
% 
fprintf('...\n\n');
for i=1:size(model_scores, 1)
    for j=1:size(model_scores, 1)
        if(j<=i)
            continue;
        end
        x1 = model_scores{i, 1};
        x2 = model_scores{j, 1};
        similarity = sum(x1.*x2) / ((x1*x1')^0.5 * (x2*x2')^0.5);
        fprintf('similarity of model %d %d is %f\n', i, j, similarity);
        
    end
end
fprintf('...\n\n');
end

