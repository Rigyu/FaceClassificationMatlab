%%
faceGallery = imageSet('PeopleFace','recursive');
galleryNames = {faceGallery.Description};
displayFaceGallery(faceGallery, galleryNames);

%%
queryFileName = 'images.jpg';
queryFace = imread(fullfile('facequery',queryFileName));
figure
imshow(queryFace)
title ('Query Image')

%%
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
bbox = step(faceDetector, queryFace);
queryFaceDetected = insertShape(queryFace,'rectangle',bbox,'LineWidth',5);
figure; imshow(queryFaceDetected); title('Detected face');

%%
queryFace = imcrop(queryFace, [bbox(1) bbox(2)-50 bbox(3)+100 bbox(4)+100]);
scaleFactor = 150/size(queryFace, 1);
queryFace = imresize(queryFace, scaleFactor);
figure; imshow(queryFace); title('Cropped-normalized face');

%%
tic
[features, locations] = helperExtractDenseSURF