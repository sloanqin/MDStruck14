function plot_scores_map(examples, scores, figure_ind)
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

figure(4);
subplot(2, 2, figure_ind);

%[scores,idx] = sort(scores,'descend');
confidence = max(scores);
%fprintf('model %d, confi %f, max_z %f \n', figure_ind, confidence,max(max(Z)));
scores = scores - min(scores) + 1e-8;
scores = scores/sum(scores);
entropy = -scores*log(scores)';
confi_norm = max(scores)*1000;

center_x = examples(:,1) + double(examples(:,3))/2;
center_y = examples(:,2) + double(examples(:,4))/2;
z = scores(:,:);

xlin = linspace(min(center_x),max(center_x),200);
ylin= linspace(min(center_y),max(center_y),200);
[X,Y] = meshgrid(xlin,ylin);
Z = griddata(center_x,center_y,z,X,Y,'v4');



% draw picture
mesh(X,Y,Z);
contour3(X,Y,Z,30);
axis tight; hold on;
%plot3(center_x,center_y,z,'.','MarkerSize',15);
title(['model ' num2str(figure_ind) ',confi ' num2str(confidence) ',entropy ' num2str(entropy) ]);
title(['model ' num2str(figure_ind) ',confi ' num2str(confidence) ',confinorm ' num2str(confi_norm) ]);
hold off;
end

