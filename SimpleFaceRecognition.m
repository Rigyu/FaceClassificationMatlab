%% Simple Face Recognition Example
%  Copyright 2014-2015 The MathWorks, Inc.

%% Load Image Information from ATT Face Database Directory
faceDatabase = imageSet('PeopleFace','recursive');

%% Display Montage of First Face
figure;
montage(faceDatabase(1).ImageLocation);
title('Images of Single Face');

%%  Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
for i=1:size(faceDatabase,2)
imageList(i) = faceDatabase(i).ImageLocation(5);
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
diff = zeros(1,9);

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2]);

%% Extract and display Histogram of Oriented Gradient Features for single face
person = 9;
[hogFeature, visualization]= ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(1,2,1);imshow(read(training(person),1));title('Input Face');
subplot(1,2,2);plot(visualization);title('HoG Feature');

%%export_fig HoGFeatures3.png used only on local to make figs

%% Extract HOG Features for training set
trainingFeatures = zeros(size(training,2)*training(1).Count,size(hogFeature,2));
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
%% Test Images from Test Set
person = 1;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
% Map back to training set to find identity
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

%% Test 2 times people from Test Set
figure;
figureNum = 1;
for person=1:12
    %for j = 1:test(person).Count
     for j = 10:11 %chose sample number
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);

        subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');

        figureNum = figureNum+2;
        %%export_fig(sprintf('Resultados/plot%d.png', person)); used only
        %%local to save images for report.
  end
    figure;
    figureNum = 1;
end
