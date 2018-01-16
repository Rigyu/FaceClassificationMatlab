%% Display Face Gallery
%Copyright 2014-2015 The MathWorks, Inc. 
function displayFaceGallery(faceGallery,galleryNames)

% displaying all the gallery images
figure
for i = 1:length(faceGallery)
    I = cell(1, faceGallery(i).Count);
    % concatenate all the images of a person side-by-side
    for j = 1:faceGallery(i).Count
        image = read(faceGallery(i), j);        
        % scale images to have same height, maintaining the aspect ratio
        scaleFactor = 150/size(image, 1); 
        image = imresize(image, scaleFactor);        
        I{j} = image;
    end   
    subplot(length(faceGallery), 1, i);
    imshow(cell2mat(I));
    title(galleryNames{i}, 'FontWeight', 'normal');
end

annotation('textbox', [0 0.9 1 0.1], 'String', 'Face Gallery', ...
     'EdgeColor', 'none', 'FontWeight', 'bold', ...
     'FontSize', 12, 'HorizontalAlignment', 'center')
