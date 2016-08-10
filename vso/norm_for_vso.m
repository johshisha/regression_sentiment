
function [XX] = norm_for_vso(X)
% X(img_idx, :)=[gLBP' gBOW' gGist' gAtt' gColor' gBOW2']; 
% gEarly=[gLBP' gBOW' gGist' gAtt' gColor'];

n = size(X, 1);

dim_bow = 500;
dim_bow2 = 1500;
dim_gist = 320;
dim_color = 768;
dim_att = 2000;
dim_lbp = 59;
dim = dim_bow2 + dim_gist + dim_color + dim_att + dim_lbp;

dim1 = 1;
dim2 = dim_lbp;
LBP = normr(X(:, dim1:dim2));

dim1 = 1+dim2;
dim2 = dim1+dim_bow-1;

dim1 = 1+dim2;
dim2 = dim1+dim_gist-1;
Gist = normr(X(:, dim1:dim2));

dim1 = 1+dim2;
dim2 = dim1+dim_att-1;
Att = normr(X(:, dim1:dim2));

dim1 = 1+dim2;
dim2 = dim1+dim_color-1;
Color = normr(X(:, dim1:dim2));

dim1 = 1+dim2;
dim2 = dim1+dim_bow2-1;
BOW2 = normr(X(:, dim1:dim2));

XX = [LBP BOW2 Gist Att Color];