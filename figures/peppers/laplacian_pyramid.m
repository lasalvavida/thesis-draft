dev = 2;

peppers = imread('./peppers.png');
peppers_x2 = imgaussfilt(peppers, dev);
peppers_x2 = imresize(peppers_x2, 0.5);
imwrite(peppers_x2, './peppers_x2.png');
l_peppers = peppers - imresize(peppers_x2, 2);
imwrite(l_peppers, './l_peppers.png');

peppers_x4 = imgaussfilt(peppers_x2, dev);
peppers_x4 = imresize(peppers_x4, 0.5);
imwrite(peppers_x4, './peppers_x4.png');
l_peppers_x2 = peppers_x2 - imresize(peppers_x4, 2);
imwrite(l_peppers_x2, './l_peppers_x2.png');

peppers_x8 = imgaussfilt(peppers_x4, dev);
peppers_x8 = imresize(peppers_x8, 0.5);
imwrite(peppers_x8, './peppers_x8.png');
l_peppers_x4 = peppers_x4 - imresize(peppers_x8, 2);
imwrite(l_peppers_x4, './l_peppers_x4.png');

peppers_blur_left = imread('./peppers_blur_left.png');
l_peppers_blur_left = peppers_blur_left - imresize(imresize(imgaussfilt(peppers_blur_left, dev), 0.5), 2);
imwrite(l_peppers_blur_left, './l_peppers_blur_left.png');

peppers_blur_right = imread('./peppers_blur_right.png');
l_peppers_blur_right = peppers_blur_right - imresize(imresize(imgaussfilt(peppers_blur_right, dev), 0.5), 2);
imwrite(l_peppers_blur_right, './l_peppers_blur_right.png');

peppers_fused_naive = uint8(zeros(size(peppers)));
for i = 1:size(peppers_fused_naive, 1)
    for j = 1:size(peppers_fused_naive, 2)
        if mean(l_peppers_blur_left(i,j)) >= mean(l_peppers_blur_right(i,j))
            peppers_fused_naive(i,j,:) = peppers_blur_left(i,j,:);
        else
            peppers_fused_naive(i,j,:) = peppers_blur_right(i,j,:);
        end
    end
end
imwrite(peppers_fused_naive, './peppers_fused_naive.png');

peppers_fused = uint8(zeros(size(peppers)));
for i = 1:size(peppers_fused, 1)
    for j = 1:size(peppers_fused, 2)
        left = mean(l_peppers_blur_left(i,j)) + 1;
        right = mean(l_peppers_blur_right(i,j)) + 1;
        peppers_fused(i,j,:) = (left/(left + right)) * peppers_blur_left(i,j,:) + (right/(left + right)) * peppers_blur_right(i,j,:);
    end
end
imwrite(peppers_fused, './peppers_fused.png');