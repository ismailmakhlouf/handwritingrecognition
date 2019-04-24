%
% PROTOTYPE 
% function pca (path, trainList, subDim) 
%  
% USAGE EXAMPLE(S) 
% pca ('C:/FERET_Normalised/', trainList500Imgs, 200); 
% 
% GENERAL DESCRIPTION 
% Implements the standard Turk-Pentland Eigenfaces method. As a final 
% result, this function saves pcaProj matrix to the disk with all images 
% projected onto the subDim-dimensional subspace found by PCA. 
%  
% REFERENCES 
% M. Turk, A. Pentland, Eigenfaces for Recognition, Journal of Cognitive 
% Neurosicence, Vol. 3, No. 1, 1991, pp. 71-86 
%  
% M.A. Turk, A.P. Pentland, Face Recognition Using Eigenfaces, Proceedings 
% of the IEEE Conference on Computer Vision and Pattern Recognition, 
% 3-6 June 1991, Maui, Hawaii, USA, pp. 586-591 
% 
% All references available on http://www.face-rec.org/algorithms/ 
%  
% INPUTS: 
% path          - full path to the normalised images from FERET database 
% trainList     - list of images to be used for training. names should be 
%                 without extension and .pgm will be added automatically 
% subDim        - Numer of dimensions to be retained (the desired subspace 
%                 dimensionality). if this argument is ommited, maximum 
%                 non-zero dimensions will be retained, i.e. (number of training images) - 1 
% 
% OUTPUTS: 
% Function will generate and save to the disk the following outputs: 
% DATA          - matrix where each column is one image reshaped into a vector 
%               - this matrix size is (number of pixels) x (number of images), uint8 
% imSpace       - same as DATA but only images in the training set 
% psi           - mean face (of training images) 
% zeroMeanSpace - mean face subtracted from each row in imSpace 
% pcaEigVals    - eigenvalues 
% w             - lower dimensional PCA subspace 
% pcaProj       - all images projected onto a subDim-dimensional space 
% 
% NOTES / COMMENTS 
% * The following files must either be in the same path as this function 
%   or somewhere in Matlab's path: 
%       1. listAll.mat - containing the list of all 3816 FERET images 
% 
% ** Each dimension of the resulting subspace is normalised to unit length 
% 
% *** Developed using Matlab 7 
% 
% 
% REVISION HISTORY 
% - 
%  
% RELATED FUNCTIONS (SEE ALSO) 
% createDistMat, feret 
%  
% ABOUT 
% Created:        03 Sep 2005 
% Last Update:    - 
% Revision:       1.0 
%  
% AUTHOR:   Kresimir Delac 
% mailto:   kdelac@ieee.org 
% URL:      http://www.vcl.fer.hr/kdelac 
% 
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE 
% OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PAPER: 
% Delac K., Grgic M., Grgic S., Independent Comparative Study of PCA, ICA, and LDA  
% on the FERET Data Set, International Journal of Imaging Systems and Technology, 
% Vol. 15, Issue 5, 2006, pp. 252-260 
% 
 
 
 
% If subDim is not given, n - 1 dimensions are 
% retained, where n is the number of training images 

% Performing PCA on Training Set
clear
% path = 'C:\Users\Ahmed\Documents\CS585\Characters - Copy (2)\';
path = 'I:\Project\Characters - Copy (2)\';
listing = dir(fullfile(path, '*.jpg'));
sum = {};

for i=1:numel(listing)
   filenames = listing(i).name;

sum{i} = filenames;
end

sum = transpose(sum);

clear filenames;

 
disp(' ') 
 
% load listAll; 

listAll = sum;
trainList = listAll;
 
% Constants 
numIm = length(trainList); 
subDim = length(trainList);
 
 
% Memory allocation for DATA matrix 
fprintf('Creating DATA matrix\n') 
tmp = imread ( [path char(listAll(1))] ); 
% [m, n] = size (tmp);                    % image size - used later also!!! 
m2 = 64;
n2 = 64;
m = 95;
thetamax = 180;
n = thetamax+1;
DATA = uint8 (zeros(m*n, numIm));       % Memory allocated 
clear str tmp; 
 
% Creating DATA matrix 
for i = 1 : numIm 
    im2 = imread ( [path char(listAll(i))] );
    im2 = rgb2gray(im2);

    im = imresize(im2, [m2 n2]);
    im3 = medfilt2(im);
    figure;
%     imshow(im);
    theta = 0:thetamax;
    [R,xp] = radon(im3,theta);
    imshow(R,[],'Xdata',theta,'Ydata',xp,...
            'InitialMagnification','fit')
    xlabel('\theta (degrees)')
    ylabel('x''')
    colormap(hot), colorbar
    iptsetpref('ImshowAxesVisible','off')
    R2 = imresize(R, [m n]);
    DATA(:, i) = reshape (R2, m*n, 1); 
end; 
save DATA DATA; 
clear im; 
clear R2;

% Creating training images space 
fprintf('Creating training images space\n') 
dim = length (trainList); 
imSpace = zeros (m*n, dim); 
for i = 1 : dim 
    index = strmatch (trainList(i), listAll); 
    imSpace(:, i) = DATA(:, index); 
end; 
save imSpace imSpace; 
clear DATA; 
 
% Calculating mean face from training images 
fprintf('Zero mean\n') 
psi = mean(double(imSpace'))'; 
save psi psi; 
 
% Zero mean 
zeroMeanSpace = zeros(size(imSpace)); 
for i = 1 : dim 
    zeroMeanSpace(:, i) = double(imSpace(:, i)) - psi; 
end; 
save zeroMeanSpace zeroMeanSpace; 
clear imSpace; 
 
% PCA 
fprintf('PCA\n') 
L = zeroMeanSpace' * zeroMeanSpace;         % Turk-Pentland trick (part 1) 
[eigVecs, eigVals] = eig(L); 
 
diagonal = diag(eigVals); 
[diagonal, index] = sort(diagonal); 
index = flipud(index); 
  
pcaEigVals = zeros(size(eigVals)); 
for i = 1 : size(eigVals, 1) 
    pcaEigVals(i, i) = eigVals(index(i), index(i)); 
    pcaEigVecs(:, i) = eigVecs(:, index(i)); 
end; 
 
pcaEigVals = diag(pcaEigVals); 
pcaEigVals = pcaEigVals / (dim-1); 
pcaEigVals = pcaEigVals(1 : subDim);        % Retaining only the largest subDim ones 
 
pcaEigVecs = zeroMeanSpace * pcaEigVecs;    % Turk-Pentland trick (part 2) 
 
save pcaEigVals pcaEigVals; 
 
% Normalisation to unit length 
fprintf('Normalising\n') 
for i = 1 : dim 
    pcaEigVecs(:, i) = pcaEigVecs(:, i) / norm(pcaEigVecs(:, i)); 
end; 
 
% Dimensionality reduction.  
fprintf('Creating lower dimensional subspace\n') 
w = pcaEigVecs(:, 1:subDim); 
save w w; 
clear w; 
 
% Subtract mean face from all images 
load DATA; 
load psi; 
zeroMeanDATA = zeros(size(DATA)); 
for i = 1 : size(DATA, 2) 
    zeroMeanDATA(:, i) = double(DATA(:, i)) - psi; 
end; 
clear psi; 
clear DATA; 
 
% Project all images onto a new lower dimensional subspace (w) 
fprintf('Projecting all images onto a new lower dimensional subspace\n') 
load w; 
pcaProj = w' * zeroMeanDATA; 
clear w; 
clear zeroMeanDATA; 
save pcaProj pcaProj;

% contents = dir(fullfile(path, '*.jpg'));
% count = 0;
% sum =zeros(0, 0);
% 
% for i = 1:numel(contents)
%     filename = contents(i).name;
%     % Open the file specified in filename, do your processing...
%     imgPath = strcat(path,'\',filename);
%     A = imread(imgPath);
%    % image(A);
%     A = reshape(A,64*64,1);
%    % display(imgPath);
%     sum = [sum A];
%     count = count +1;
%     %display(sum);
% end
% %display(sum);
% %[coeff,score,latent] = princomp(sum)
% % path = 'C:\Users\Ahmed\Documents\CS585\Characters\As.jpg';
% % im2 = imread(path);
% % load psi;
% %     im2 = rgb2gray(im2);
% % 
% %     im = imresize(im2, [m n]);
% % image = reshape (im, m*n, 1);    
% % zeroMeanDATA1 = double(image) - psi; 
% % load w;
% % pcaProjImage = w' * zeroMeanDATA1;
% % sumD = zeros(1,numel(listing));
% % 
% % for i = 1 :  numel(listing)
% %     
% %     for j = 1  :   subDim
% %         
% %         difference(j,i) =  pcaProj(j,i) - pcaProjImage(j); 
% %         
% %     end
% %     
% % end
% % % for i = 1 : numel(listing)
% % %     difference(:,i) = norm(pcaProj(:,i) - pcaProjImage);
% % % end
% % for j = 1 :  numel(listing)
% %     for i = 1 : subDim
% %         sumD(i) = sumD(i) + difference(i,j);
% %     end
% % end
% % 
% % 
% % 
% % [C,I] = min(sumD);
% % display(I);
% % path = 'C:\Users\Ahmed\Documents\CS585\PCA\Faces Gray Dataset 1 (1)\cars128x128\';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = 'I:\Project\Testing\';


listing = dir(fullfile(path, '*.jpg'));
sumx = {};

for i=1:numel(listing)
   filenamesnew = listing(i).name;
sumx{i} = filenamesnew;
end

sumx = transpose(sumx);

clear filenamesnew;
 
disp(' ') 
 
% load listAll; 

listAllnew = sumx;
testList = listAllnew;

clear sumx;
% clear listAllnew;
% Constants 
numIm = length(testList); 

 
 
% Memory allocation for DATANEW matrix 
fprintf('Creating DATANEW matrix\n') 
tmp = imread ( [path char(listAllnew(1))] ); 
% [m, n] = size (tmp);                    % image size - used later also!!! 
% m = 64;
% n = 64;
DATANEW = uint8 (zeros(m*n, numIm));       % Memory allocated 
clear str tmp; 
 
% Creating DATANEW matrix 
for i = 1 : numIm 
    im2 = imread ( [path char(listAllnew(i))] ); 
    im2 = rgb2gray(im2);
    im = imresize(im2, [m2 n2]);
    im3 = medfilt2(im);
     theta = 0:thetamax;
    [R,xp] = radon(im3,theta);
    R2 = imresize(R, [m n]);
    DATANEW(:, i) = reshape (R2, m*n, 1); 
end
fprintf('Error') 
save DATANEW DATANEW; 
clear im; 
clear R2;




% Subtract mean face from new image(s) 
load DATANEW; 
load psi; 
zeroMeanDATAnew = zeros(size(DATANEW)); 
for i = 1 : size(DATANEW, 2) 
    zeroMeanDATAnew(:, i) = double(DATANEW(:, i)) - psi; 
end; 
% clear psi; 
% clear DATANEW; 

% Project all images onto a new lower dimensional subspace (w) 
fprintf('Projecting new image(s) onto a new lower dimensional subspace\n') 
load w; 
pcaProjnew = w' * zeroMeanDATAnew; 
% clear w; 
% clear zeroMeanDATAnew; 
save pcaProjnew pcaProjnew;

% Euclidean distance 

load pcaProj;
load pcaProjnew;
load w;
load imSpace;

reconzero = zeros(size(imSpace)); 
recon = zeros(size(imSpace));
reconzero = w*pcaProj;

for i = 1 : dim 
    recon(:, i) = double(reconzero(:, i)) + psi; 
end; 


reconzeronew = w*pcaProjnew;

fprintf('Training Reconstruction complete\n');

for i = 1 : size(testList, 1)
reconnew(:, i) = double(reconzeronew(:,i)) + psi; 
end;

fprintf('Testing Reconstruction complete\n');


% diff = zeros(size(recon));
% sumdiff2 = zeros(size(reconnew, 2), size(recon, 2));
% sqrtdiff2 = zeros(size(reconnew, 2), size(recon, 2));
% for k = 1: size(reconnew, 2)
%     for i = 1 : size(recon, 2) 
%         for j = 1 : size(reconnew, 1) 
%         diff(j, i) = (reconnew(j,k) - recon(j, i))^2;
%         sumdiff2(k, i) = sumdiff2(k, i) + diff(j, i);
%         end;
%         sqrtdiff2(k, i) = sqrt(sumdiff2(k, i));
% 
%     end;
diff = zeros(size(pcaProj));
sumdiff2 = zeros(size(pcaProjnew, 2), size(pcaProj, 2));
sqrtdiff2 = zeros(size(pcaProjnew, 2), size(pcaProj, 2));
for k = 1: size(pcaProjnew, 2)
    for i = 1 : size(pcaProj, 2) 
        for j = 1 : size(pcaProjnew, 1) 
        diff(j, i) = (pcaProjnew(j,k) - pcaProj(j, i))^2;
        sumdiff2(k, i) = sumdiff2(k, i) + diff(j, i);
        end;
        sqrtdiff2(k, i) = sqrt(sumdiff2(k, i));

    end;
fprintf('Differences calculated \n')
end;

[possmallsumdiff, index] = min(sqrtdiff2, [],2);
display(possmallsumdiff);
posmaxsumdiff = max(possmallsumdiff);
posminsumdiff = min(possmallsumdiff);
posavgsumdiff = mean(possmallsumdiff);
display(posmaxsumdiff);
display(posminsumdiff);
display(posavgsumdiff);
display(index);
display(listAll(index));

clear diff;
clear sumdiff2;
clear sqrtdiff2;
% 
% 
% 
% % Negative Testing 
% path = 'C:\Users\Ahmed\Documents\CS585\PCA\Faces Gray Dataset 1 (1)\cars128x128_negative_testing\';
% 
% 
% listing = dir(fullfile(path, '*.jpg'));
% 
% x = length(listing);
% sumx = {};
% 
% for i=1:numel(listing)
%    filenamesnew = listing(i).name;
% sumx{i} = filenamesnew;
% end
% 
% sumx = transpose(sumx);
% 
% clear filenamesnew;
% 
%  
% disp(' ') 
% 
% 
% listAllnew = sumx;
% testList = listAllnew;
% 
% clear sumx;
% 
% numIm = length(testList); 
% 
%  
% % Memory allocation for DATANEW matrix 
% fprintf('Creating DATANEW matrix\n') 
% tmp = imread ( [path char(listAllnew(1))] ); 
% % [m, n] = size (tmp);                    % image size - used later also!!! 
% m = 64;
% n = 64;
% DATANEW = uint8 (zeros(m*n, numIm));       % Memory allocated 
% clear str tmp; 
%  
% % Creating DATANEW matrix 
% for i = 1 : numIm 
%     im2 = imread ( [path char(listAllnew(i))] ); 
%     im2 = rgb2gray(im2);
%     im = imresize(im2, [m n]);
%     DATANEW(:, i) = reshape (im, m*n, 1); 
% end; 
% save DATANEW DATANEW; 
% clear im; 
% 
% % Subtract mean face from new image(s) 
% load DATANEW; 
% load psi; 
% zeroMeanDATAnew = zeros(size(DATANEW)); 
% for i = 1 : size(DATANEW, 2) 
%     zeroMeanDATAnew(:, i) = double(DATANEW(:, i)) - psi; 
% end; 
% 
% 
% fprintf('Projecting new image(s) onto a new lower dimensional subspace\n') 
% load w; 
% pcaProjnew = w' * zeroMeanDATAnew; 
% 
% save pcaProjnew pcaProjnew;
% 
% 
% load pcaProj;
% load pcaProjnew;
% load w;
% 
% 
% reconzero = zeros(size(imSpace)); 
% recon = zeros(size(imSpace));
% reconzero = w*pcaProj;
% for i = 1 : dim 
%     recon(:, i) = double(reconzero(:, i)) + psi; 
% end; 
% 
% % reconzeronew = zeros(size(psi)); 
% % reconnew = zeros(size(psi));
% reconzeronew = w*pcaProjnew;
% 
% fprintf('Training Reconstruction complete\n');
% 
% for i = 1 : size(testList, 1)
% reconnew(:, i) = double(reconzeronew(:,i)) + psi; 
% end;
% 
% fprintf('Testing Reconstruction complete\n');
% 
% % Works - Produces undesirable results
% diff = zeros(size(recon));
% sumdiff2 = zeros(size(reconnew, 2), size(recon, 2));
% sqrtdiff2 = zeros(size(reconnew, 2), size(recon, 2));
% for k = 1: size(reconnew, 2) % 31
%     for i = 1 : size(recon, 2) % 95
%         for j = 1 : size(reconnew, 1)  % 4096
%         diff(j, i) = (reconnew(j,k) - recon(j, i))^2;
%         sumdiff2(k, i) = sumdiff2(k, i) + diff(j, i);
%         end;
%         sqrtdiff2(k, i) = sqrt(sumdiff2(k, i));
% 
%     end;
% 
% end;
% [negsmallsumdiff, index] = min(sqrtdiff2, [],2);;
% display(negsmallsumdiff);
% negmaxsumdiff = max(negsmallsumdiff);
% negminsumdiff = min(negsmallsumdiff);
% negavgsumdiff = mean(negsmallsumdiff);
% display(negmaxsumdiff);
% display(negminsumdiff);
% display(negavgsumdiff);
% display(index);
% display(listAll(index));
% 
% theta = (posavgsumdiff + negavgsumdiff)/2;
% display(theta);
% 
% maxsize = numIm + dim;
% 
% 
% 
% hits = 1;
% misses = 1;
% falsehits = 1;
% truemisses = 1;
% 
% for theta2 = posminsumdiff: 0.1e+03: negmaxsumdiff
%     for i = 1: size(possmallsumdiff, 1)
%         if (possmallsumdiff(i) <= theta2)
% 
%                 hits = hits + 1;
% 
% 
%         elseif (possmallsumdiff(i) > theta2)
% 
%                 misses = misses + 1;
% 
%         end
% 
%         if (negsmallsumdiff(i) <= theta2)
% 
%                 falsehits = falsehits + 1;
% 
% 
%         elseif (possmallsumdiff(i) > theta2)
% 
%                 truemisses = truemisses + 1;
% 
%         end
% 
%     end;
% 
% 
%     ROC(falsehits, hits) = hits;
% end
% 
% plot(ROC, 'x');                
% title('ROC Curve');
% xlabel('False Alarms');
% ylabel('Hits');
% 
% reconImage = recon(:,1);
% %im = imresize(im2, [m n]);
% image1111 = reshape (reconImage, m, n); 
% figure;
% image(image1111);