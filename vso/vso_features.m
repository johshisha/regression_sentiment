
function [VSO] = vso_features(X_path, out_VSO_path)

addpath('vso1.1');
addpath ('vlfeat-0.9.20/toolbox');
run vl_setup;

load('classes.mat');

load(X_path);
X = norm_for_vso(X);

% n:サンプルサイズ 
% dx:次元
[n, dx] = size(X);

models = {};
classnum = length(classes);
for tempj=1:classnum
    currentClass=classes{tempj};
    try
        filename = sprintf('vso1.1/SVM/%s.mat', currentClass);
        load(filename);
    catch exception
        disp('Error');
    end
    models{tempj} = modelfusion;
end

VSO = zeros(n, classnum);
for tempj=1:classnum
    disp(tempj);
    [predict_label, accuracy, dec_values] = predict(ones(n, 1), sparse(X), models{tempj},'-b 1');
    VSO(:, tempj) = dec_values(:, 1)';
end
save(out_VSO_path, 'VSO');

%{
targetFile='testEarlyFusion';
label=1;
fileID = fopen(targetFile,'a+');
for i= 1:10
    fprintf(fileID,'%d',label);
    fprintf(fileID,' ');
    for j = 1:dx
        fprintf(fileID,'%d',j);
        fprintf(fileID,':');
        if(isnan(X(i, j)))
            fprintf(fileID,'%f',0);
        else
            fprintf(fileID,'%f', X(i, j));
        end
        fprintf(fileID,' ');
    end
    fprintf(fileID,'\n');
end
fclose(fileID);
[label_descriptortest, inst_descriptortest] = libsvmread('testEarlyFusion');

for tempj=1:classnum
    [predict_label, accuracy, dec_values] = predict(label_descriptortest, inst_descriptortest, modelfusion,'-b 1');
    biconceptVector(tempj,1) = dec_values(1);
end

%}


