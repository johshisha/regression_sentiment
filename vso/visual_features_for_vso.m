
% 算出した特徴は，使う前に正規化すること！

addpath('vso1.1');
addpath('vlfeat-0.9.20/toolbox');
run vl_setup;

img_list = 'icassp_dataset/insta_image_list1.txt';
fin = fopen(img_list);
img_names = textscan(fin, '%s %s');
fclose(fin);
out_feat_mat = 'icassp_dataset/insta_X1';
load('model.mat');
load('model_2000.mat');

% 算出する特徴の次元
dim_bow = 500;
dim_bow2 = 1500;
dim_gist = 320;
dim_color = 768;
dim_att = 2000;
dim_lbp = 59;
dim = dim_bow + dim_gist + dim_color + dim_att + dim_lbp + dim_bow2;
% 画像枚数
N = size(img_names{1}, 1);
N = 50000;
i = 1;


% LBP用
mapping=getmapping(8,'u2');

% 特徴行列のサイズ確保
X = zeros(N, dim);

for img_idx = i:1:(i+N-1),
    % 画像読み込み
    img_name = sprintf('/Users/marie-pro/Data/instagram/images/%s.jpg', img_names{2}{img_idx});
    try
        im = imread(img_name);
    catch
        disp('Failed to read image!');
        img_name  
        continue;
    end

    [w,h,c] = size(im);
    if (w<10||h<h)
        disp('Image too small!');
        continue;
    end

    if (c == 1)
        im = gray2rgb(im);
    end

    %%%%%%%%%% BOW features %%%%%%%%%%%%
    %disp('BOW');
    %tic
    k1=getImageDescriptor(model, im);
    k3=cat(2,k1);
    %gBOW= vl_homkermap(k3, 1, 'kchi2', 'gamma', .5);
    gBOW = k3;
    gBOW2 = vl_homkermap(k3, 1, 'kchi2', 'gamma', .5);
    %toc
    %%%%%%%%%% GIST features %%%%%%%%%%%
    %disp('GIST');
    %tic
    img = rgb2gray(im);
    imgsize = size(img);
    lengthTemp = imgsize(1);
    widthTemp = imgsize(2);

    if lengthTemp > widthTemp
        img=img(int32((lengthTemp-widthTemp)/2+1):int32((lengthTemp+widthTemp)/2),:);
    else
        img=img(:,int32((widthTemp-lengthTemp)/2+1):int32((lengthTemp+widthTemp)/2));
    end
    imgsize = size(img);
    lengthTemp = imgsize(1);
    widthTemp = imgsize(2);
    img = img(1:min(lengthTemp,widthTemp),1:min(lengthTemp,widthTemp));

    imgsize = size(img);
    lengthTemp = imgsize(1);
    widthTemp = imgsize(2);

    % Parameters:
    Nblocks = 4;
    imageSize = min(lengthTemp,widthTemp);
    orientationsPerScale = [8 8 4];
    numberBlocks = 4;

    % Precompute filter transfert functions (only need to do this one, unless image size is changes):
    createGabor(orientationsPerScale, imageSize); % this shows the filters
    G = createGabor(orientationsPerScale, imageSize);

    % Computing gist requires 1) prefilter image, 2) filter image and collect
    % output energies
    output = prefilt(double(img), 4);
    gGist = gistGabor(output, numberBlocks, G);
    %toc
    %%%%%%%%%% Color features %%%%%%%%%%%
    %disp('color');
    %tic
    I = im;
    nBins = 256;
    rHist = imhist(I(:,:,1), nBins);
    gHist = imhist(I(:,:,2), nBins);
    bHist = imhist(I(:,:,3), nBins);
    gColor = [rHist;gHist;bHist];
    %toc
    %%%%%%%%%% LBP features %%%%%%%%%%%%%%%
    %disp('LBP');
    img = im;
    %mapping=getmapping(8,'u2');
    g=lbp(img,1,8,mapping,'h');
    gLBP=g';

    %%%%%%%%%%% Attribute features %%%%%%%%%
    %disp('Att');
    %tic
    att = att_ext(normc(gBOW2)', normc(gLBP)',  normc(gGist)', normc(gColor)', model_all);
    gAtt = att';
    %toc
       
    X(img_idx, :)=[gLBP' gBOW' gGist' gAtt' gColor' gBOW2']; 

    if (mod(img_idx, 1000)==1),
        fprintf('%d\n', img_idx);
    end
    im = 0;
    img = 0;
end

save(out_feat_mat, 'X', '-v7.3');
