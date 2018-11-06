function [target_loc, model_ind] = choose_target_loc(target_locs)
% plot_scores_map
% plot 3D map of examples' scores
%
% INPUT:
%   examples  - parameters of structured svm: support patterns and support vectors
%   scores  - fc4 features and example boxes
%   figure_ind  - index of figure to plot
%
% OUTPUT:
%   model_ind  - index of model
%
% Sloan Qin, 2017
% 

if(size(target_locs, 1)==1)
    target_loc = target_locs{1,1};
    return;
end

dist_matrix = zeros(size(target_locs, 1), size(target_locs, 1));
for i=1:size(target_locs, 1)
    for j=i:size(target_locs, 1)
        if(i==j)
            dist_matrix(i, j) = -1.0;
            dist_matrix(j, i) = -1.0;
            continue;
        end
        box_i = target_locs{i, 1};
        box_j = target_locs{j, 1};
        center_x_i = box_i(1,1) + box_i(1,3)/2;
        center_y_i = box_i(1,2) + box_i(1,4)/2;
        center_x_j = box_j(1,1) + box_j(1,3)/2;
        center_y_j = box_j(1,2) + box_j(1,4)/2;
        dist_matrix(i, j) = ( (center_x_i - center_x_j)^2 + (center_y_i - center_y_j)^2 );
        dist_matrix(j, i) = dist_matrix(i, j);
    end
end

[dd, idx] = sort(dist_matrix);

votes_matrix = zeros(size(target_locs, 1), 1);
votes2add = zeros(size(target_locs, 1), 1);
votes_num = linspace(size(target_locs, 1), 1, size(target_locs, 1));
votes_num(1) = 0;
for i=1:size(target_locs, 1)
    ind = idx(:, i);
    votes2add(ind, 1) =  votes_num';
    votes_matrix = votes_matrix + votes2add; 
end

[dd, votes_idx] = sort(votes_matrix, 'descend');
target_loc = target_locs{votes_idx(1)};
model_ind = votes_idx(1);
return;

end
