function [ result ] = mdstruck_run(images, region, net, display)
% MDNET_RUN
% Main interface for MDNet tracker
%
% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%
% Hyeonseob Nam, 2015
% 

if(nargin<4), display = true; end

% declare global variables
global st_svm; 
global total_data;

%% Initialization
fprintf('Initialization...\n');

st_svms = cell(0,1);

nFrames = length(images);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
targetLoc = region;
result = zeros(nFrames, 4); result(1,:) = targetLoc;

st_svm.firstFrameTargetLoc = targetLoc;

[net_conv, net_fc, opts] = mdnet_init(img, net);

%% Train a bbox regressor
if(opts.bbreg)
    fprintf('  bbox regressor\n');
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);
    
    X = permute(gather(feat_conv),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

% extract conv3 features
feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);


%% Learning CNN
fprintf('  training cnn...\n');
net_fc = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

%% Initialize displayots
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
    
    figure(3);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
        figure(4);
    set(gcf,'Position',[200 100 800 600],'MenuBar','none','ToolBar','none');
end

%% Prepare training data for structured svm online update
%% total_data(1,1,1,:):features,total_data(1,1,2,:):y,total_data(1,1,3,:):yv
total_data = cell(1,1,3,nFrames);

examples = gen_samples('radial', targetLoc, opts.svm_samples, opts, 2, 5);

feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
%feat_fc4 = [squeeze(feat_fc4); squeeze(prep_feat_conv_data(feat_conv))];
%feat_fc4 = reshape(feat_fc4, [1,1, size(feat_fc4)]);
%feat_fc4 = prep_feat_fcX_data(net_fc, feat_conv, opts);
total_data{:,:,1,1} = double(feat_fc4(:,:,:,:));
total_data{:,:,2,1} = examples;
total_data{:,:,3,1} = examples - repmat(targetLoc,[size(examples,1),1]);

%% for debug
%{
y = importdata('./dataset/debug/y.txt',',');
y_rela = importdata('./dataset/debug/yrela.txt',',');
feat = importdata('./dataset/debug/feat.txt',',');
feat = feat';
total_data{:,:,1,1} = reshape(feat,[1,1,192,81]);
total_data{:,:,2,1} = y;
total_data{:,:,3,1} = y_rela;
%}

%% st_svm initialise
mdstruck_init();

%% structured svm update
st_svm.x_ind = 0;
st_svm.targetScores = [st_svm.targetScores;1.0];
st_svm_update(1, true);
first_frame_st_svm = st_svm;
st_svms{1,1} = st_svm;
st_svms{2,1} = st_svm;

success_frames = 1;
trans_f = opts.trans_f;	
scale_f = opts.scale_f;

%% Main loop
update_cnt = 10;
cur_model_contin_miss_cnt = 0;
target_score = 1;
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... \n', To, nFrames);
	fprintf('supportPatterns/supportVectors is %d/%d... \n', size(st_svm.supportPatterns,1), size(st_svm.supportVectors,1));
    
    st_svm.x_ind = To;
    
    %% add new st_svm
    interval = max(10, min(st_svm.snapshot_interval, round(To/4)));
    if(target_score>st_svm.update_target_score_threshold )
        update_cnt = update_cnt  - 1;
    end
    if(update_cnt==0)
        update_cnt = interval;
        st_svms = [st_svms; st_svm];
        if(size(st_svms,1)>4)
            st_svms = st_svms(2:end,1);
        end
    end
    st_svms{end,1} = st_svm;
    
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    spf = tic;
    %% Prepare Estimation data
    % draw target candidates
    mdnet_features_convX_Time = tic;
    %examples = gen_samples('pixel', targetLoc, opts.svm_eval_samples, opts, trans_f, scale_f);
    if(target_score>st_svm.update_target_score_threshold )
        examples = gen_samples('gaussian', targetLoc, opts.nSamples, opts, st_svm.trans_f, st_svm.scale_f);
    else
        examples = gen_samples('gaussian', targetLoc, opts.nSamples*3, opts, st_svm.trans_f, st_svm.scale_f);
    end
    feat_conv = mdnet_features_convX(net_conv, img, examples, opts); 
	feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
    %feat_fc4 = [squeeze(feat_fc4); squeeze(prep_feat_conv_data(feat_conv))];
    %feat_fc4 = reshape(feat_fc4, [1,1, size(feat_fc4)]);
    %feat_fc4 = prep_feat_fcX_data(net_fc, feat_conv, opts);
    total_data{:,:,1,To} = double(feat_fc4(:,:,:,:));
    total_data{:,:,2,To} = examples;
    total_data{:,:,3,To} = examples - repmat(targetLoc,[size(examples,1),1]); 
    mdnet_features_convX_Time = toc(mdnet_features_convX_Time);
    fprintf('mdnet_features_convX_Time %f seconds\n',mdnet_features_convX_Time);
   
    %% evaluate the candidates
    st_svm_eval_Time = tic;
    targetLocs = cell(size(st_svms,1), 1);
    target_scores = cell(size(st_svms,1), 1);
    model_scores = cell(size(st_svms,1), 1);
    model_idxs = cell(size(st_svms,1), 1);
    for i=1:size(st_svms,1)
        st_svm = st_svms{i,1};
        [ svs_feats, svs_beta, kernerl_sigma, xs_feats ] = prep_eval_data( To );
        scores = st_svm_eval(svs_feats, svs_beta, kernerl_sigma, xs_feats);
        model_scores{i, 1} = scores;
        if (To>0)
            %plot_scores_map(examples, scores,i);
        end       
        %scores = st_svm_eval(To);
        [scores,idx] = sort(scores,'descend');
        target_scores{i, 1} = scores(1,1);
        new_targetLoc = examples(idx(1,1),:);
        targetLocs{i, 1} = new_targetLoc;
        model_idxs{i, 1} = idx;
    end
    [new_targetLoc, model_ind] = choose_target_loc(targetLocs);
    target_score = target_scores{model_ind, 1};
    %if(target_score>st_svm.update_target_score_threshold)
        targetLoc = new_targetLoc;
    %end
    
    %% restore model
%     if(model_ind ~= size(st_svms,1))
%         cur_model_contin_miss_cnt = cur_model_contin_miss_cnt + 1;
%     else
%         cur_model_contin_miss_cnt = 0;
%     end
%     center_distance = power( (targetLocs{size(st_svms,1), 1} - targetLocs{model_ind, 1}), 2);
%     center_distance = (center_distance(1) + center_distance(2))^0.5;
%     if(cur_model_contin_miss_cnt>=4 && center_distance>10)
%         st_svm = st_svms{model_ind,1};
%         st_svms{end,1} = st_svms{model_ind,1};
%     end
    
    fprintf('target_score is %f\n',target_score);
    fprintf('target_score is %f\n',target_score);
    fprintf('target_score is %f\n',target_score);
    fprintf('target_score is %f\n',target_score);
    fprintf('target_score is %f\n',target_score);
    %calcu_similarity(model_scores);
    st_svm_eval_Time = toc(st_svm_eval_Time);
    fprintf('st_svm_eval_Time %f seconds\n',st_svm_eval_Time);
    fprintf('sample num is: %d \n',size(xs_feats,2));
    
    % final target
    result(To,:) = targetLoc;
    st_svm.targetScores = [st_svm.targetScores;target_score];
    
    % extend search space in case of failure
    if(target_score<st_svm.update_target_score_threshold )
        trans_f = min(2, 1.1*trans_f);
    else
        %trans_f = st_svm.trans_f;
        trans_f = max(st_svm.trans_f, trans_f/1.1);
    end
    
    %% bbox regression
    if(opts.bbreg) % if(opts.bbreg && target_score>0 ) qyy change
        st_svms{end,1} = st_svm;
        st_svm = first_frame_st_svm;
        idx = model_idxs{model_ind, 1};
        [ svs_feats, svs_beta, kernerl_sigma, ~ ] = prep_eval_data( 1 );
        xs_feats = squeeze( total_data{1,1,1,To}(:,:,:,idx(1)) );
        first_frame_score = st_svm_eval(svs_feats, svs_beta, kernerl_sigma, xs_feats);
        fprintf('first frame score is %f\n',first_frame_score);
        fprintf('first frame score is %f\n',first_frame_score);
        fprintf('first frame score is %f\n',first_frame_score);
        fprintf('first frame score is %f\n',first_frame_score);
        fprintf('first frame score is %f\n',first_frame_score);
        if(first_frame_score > st_svm.bbox_reg_score_threshold) % similar with first frame target
            X_ = permute(gather(feat_conv(:,:,:,idx(1))),[4,3,1,2]);
            X_ = X_(:,:);
            bbox_ = examples(idx(1),:);
            pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
            result(To,:) = round(mean(pred_boxes,1));
        end
        st_svm = st_svms{end,1};
    end
    
    %% Prepare training data
    if(target_score>st_svm.update_target_score_threshold )
        %examples = gen_samples('radial', targetLoc, opts.svm_samples, opts, 2, 5);
        examples = gen_samples('radial', targetLoc, opts.svm_samples, opts, 0.1, 2);

		feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
		feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
        %feat_fc4 = [squeeze(feat_fc4); squeeze(prep_feat_conv_data(feat_conv))];
        %feat_fc4 = reshape(feat_fc4, [1,1, size(feat_fc4)]);
        %feat_fc4 = prep_feat_fcX_data(net_fc, feat_conv, opts);
		total_data{:,:,1,To} = feat_fc4(:,:,:,:);
		total_data{:,:,2,To} = examples;
        total_data{:,:,3,To} = examples - repmat(targetLoc,[size(examples,1),1]); 
        
        success_frames = [success_frames, To];
    end
    
    %% structured svm update
    fprintf('target_score is %.12f\n',target_score);
    whether_process_new = false;
    if(target_score>st_svm.update_target_score_threshold )
        if(To>50 && mod(To,5)==0)
            whether_process_new = true;
        end
        if(To<=50)
            whether_process_new = true;
        end
    end
    
    st_svm_update_Time = tic;
	st_svm_update(To, whether_process_new);
    st_svm_update_Time = toc(st_svm_update_Time);
    fprintf('st_svm_update_Time %f seconds\n',st_svm_update_Time);
    
    spf = toc(spf);
    fprintf('%f seconds\n',spf);
    
    if (0)
        continue;
    end
    
    %% Display
    if display
        figure(2);%qyy
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
            
        colors = [0 1 0; 0 0 1;1 1 0;0 1 1];
        for i=1:size(targetLocs,1)-1
            rectangle('Position', targetLocs{i,1}, 'EdgeColor', colors(i,:), 'Linewidth', 3-0.5*i);
            text(10, 30+i*15, num2str(i), 'Color', colors(i,:), 'FontWeight','bold', 'FontSize',10);
        end
        rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
        radius = (result(To-1,3) + result(To-1,4))/2 * trans_f;
        left_x = result(To-1,1) + result(To-1,3)/2 - radius;
        left_y = result(To-1,2) + result(To-1,4)/2 - radius;
        rectangle('Position', [left_x,left_y,radius*2,radius*2],...
            'Curvature',[1,1],'EdgeColor', [1 0 0], 'Linewidth', 1);%search circle
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;
        
        % sort and display supportvectors
        if(1)
            continue;
        end
        supportVectors = cell2mat(st_svm.supportVectors);
        beta = reshape([supportVectors(:,1).b],[],1);
        [beta_sorted,index] = sort(beta,'descend');
        sorted_sVs = supportVectors(index);
        figure(3);
        hc = get(gca, 'Children'); delete(hc(1:end));
        for i=1:min(20,size(sorted_sVs,1))
            sv = sorted_sVs(i,1);
            strShow = ['beta:',num2str(sv.b),'  x_ind:',num2str(sv.x_ind)...
                ,'  y_ind:',num2str(sv.y_ind),'  sp_ind:',num2str(sv.sp_ind)];
            text(0.1,1.05-i*0.05,strShow,'Color','r', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 10); 
        end
        drawnow;
        
    end
end









