%% Load Faces and Models
faceSet = imageSet('PeopleFaceTest');
%% Display Faces
figure
for i = 1:3
        subplot(1,3,i);
        imshow(read(faceSet,i));
end
%% Perform Feature Extraction
descriptors = cell(1,3);
for i = 1:3
    % computing dense SURF
    [features,locations] = helperExtractDenseSURF(read(faceSet,i));
    featuresReduced = features * surfPCA.pcaCoeff;
    descriptors{i} = [featuresReduced, locations];
end

%Fisher Vector Encoding
fisherGMM = load('models/fisherGMM.mat');