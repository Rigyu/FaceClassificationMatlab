%% Face Recognition with MathWorks Faces 
% Copyright 2014-2015 The MathWorks, Inc.
clear all;
close all;
clc;

%% Read Mathworks Face Gallery
faceGallery = imageSet('MathWorksGallery', 'recursive');
galleryNames = {faceGallery.Description};
displayFaceGallery(faceGallery,galleryNames);

%% Create HoG training features from face gallery
trainingFeatures = zeros(19,10404);
featureCount = 1;

for i=1:size(faceGallery,2)
    for j = 1:faceGallery(i).Count
        sizeNormalizedImage = imresize(rgb2gray(read(faceGallery(i),j)),[150 150]);
        trainingFeatures(featureCount,:) = extractHOGFeatures(sizeNormalizedImage);
        trainingLabel{featureCount} = faceGallery(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = faceGallery(i).Description;
end

%% Create Classifier 
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Read test data
testSet = imageSet('MathWorksTestImages');
figure;
figureNum = 1;
for  i= 1: testSet.Count
    queryImage = read(testSet,i);
    queryFeatures = extractHOGFeatures(rgb2gray(queryImage));
    personLabel = predict(faceClassifier,queryFeatures);
    booleanIndex = strcmp(personLabel, personIndex);
    integerIndex = find(booleanIndex);
    subplot(5,2,figureNum);imshow(queryImage);title('Query Face');
    subplot(5,2,figureNum+1);imshow(read(faceGallery(integerIndex),1));title('Matched Class');
    figureNum = figureNum+2;
    
end