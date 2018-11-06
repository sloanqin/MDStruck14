% make the mex file for Windows system
% yuanyuan qin
% 2017

function compile()

% set the values
opts.clean                  =   false; % clean mode
opts.dryrun                 =   false; % dry run mode
opts.verbose                =   1; % output verbosity
opts.debug                  =   false; % enable debug symbols in MEX-files

% Clean
if opts.clean
    if opts.verbose > 0
        fprintf('Cleaning all generated files...\n');
    end

    cmd = fullfile(['*.' mexext]);
    if opts.verbose > 0, disp(cmd); end
    if ~opts.dryrun, delete(cmd); end

    cmd = fullfile('*.obj');
    if opts.verbose > 0, disp(cmd); end
    if ~opts.dryrun, delete(cmd); end

    return;
end

% compile flags
mex_flags = '';
if opts.verbose > 1
    mex_flags = ['-v ' mex_flags];    % verbose mex output
end
if opts.debug
    mex_flags = ['-g ' mex_flags];    % debug vs. optimized builds
end
compstr = computer;
is64bit = strcmp(compstr(end-1:end),'64');
if (is64bit)
  mex_flags = ['-largeArrayDims ' mex_flags];
end

% Compile st_svm_eval.cpp
src = 'st_svm_eval.cpp';
cmd = sprintf('mex %s %s', mex_flags, src);
if opts.verbose > 0, disp(cmd); end
if ~opts.dryrun, eval(cmd); end

% Compile st_svm_evaluate.cpp
src = 'st_svm_evaluate.cpp';
cmd = sprintf('mex %s %s', mex_flags, src);
if opts.verbose > 0, disp(cmd); end
if ~opts.dryrun, eval(cmd); end

end
