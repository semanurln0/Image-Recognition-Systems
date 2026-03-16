imagePath = 'image.jpg';
rgb = imread(imagePath);
if size(rgb, 3) == 3
    gray = rgb2gray(rgb);
else
    gray = rgb;
end

eq = histeq(gray);

edges = edge(gray, 'Canny');

blurred = imgaussfilt(gray, 1.0);

figure('Name','Image Variations','Color','w');
subplot(2,3,1); imshow(rgb);     title('Task 1: Original (imread, imshow)');
subplot(2,3,2); imshow(gray);    title('Task 2: Grayscale (rgb2gray)');
subplot(2,3,3); imshow(eq);      title('Task 3: Equalized (histeq)');
subplot(2,3,4); imshow(edges);   title('Task 4: Edges (edge)');
subplot(2,3,5); imshow(blurred); title('Task 5: Gaussian (imgaussfilt)');