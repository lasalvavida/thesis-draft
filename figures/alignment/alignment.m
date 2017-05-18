original_left = imread('waterboy_left.png');
original_right = imread('waterboy_right.png');

left = im2double(rgb2gray(original_left));
right = im2double(rgb2gray(original_right));

m = 5;
s = 1.0;

left = medfilt2(left, [m m]);
left = imgaussfilt(left, s);
right = medfilt2(right, [m m]);
right = imgaussfilt(right, s);

xx_9 = [
    0 0 0  0  0  0 0 0 0;
    0 0 0  0  0  0 0 0 0;
    1 1 1 -2 -2 -2 1 1 1;
    1 1 1 -2 -2 -2 1 1 1;
    1 1 1 -2 -2 -2 1 1 1;
    1 1 1 -2 -2 -2 1 1 1;
    1 1 1 -2 -2 -2 1 1 1;
    0 0 0  0  0  0 0 0 0;
    0 0 0  0  0  0 0 0 0
];

xy_9 = [
    0  0  0  0 0  0  0  0 0;
    0  1  1  1 0 -1 -1 -1 0;
    0  1  1  1 0 -1 -1 -1 0;
    0  1  1  1 0 -1 -1 -1 0;
    0  0  0  0 0  0  0  0 0;
    0 -1 -1 -1 0  1  1  1 0;
    0 -1 -1 -1 0  1  1  1 0;
    0 -1 -1 -1 0  1  1  1 0;
    0  0  0  0 0  0  0  0 0
];

yy_9 = xx_9';

xx_left = conv2(left, xx_9, 'same');
xy_left = conv2(left, xy_9, 'same');
yy_left = conv2(left, yy_9, 'same');

N = 9;

MAX_FEATURES = 1000;

h_left = zeros(size(left));

for i = 1:size(left, 1)
    for j = 1:size(left, 2)
        if i < N || i > size(left, 1) - N
            h_left(i,j) = 0;
        elseif j < N || j > size(left, 2) - N
            h_left(i,j) = 0;
        else
            h_left(i,j) = abs(xx_left(i,j) * yy_left(i,j) - 2 * xy_left(i,j));
        end
    end
end

xx_right = conv2(right, xx_9, 'same');
xy_right = conv2(right, xy_9, 'same');
yy_right = conv2(right, yy_9, 'same');

h_right = zeros(size(right));
for i = 1:size(right, 1)
    for j = 1:size(right, 2)
        if i < N || i > size(right, 1) - N
            h_right(i,j) = 0;
        elseif j < N || j > size(right, 2) - N
            h_right(i,j) = 0;
        else
            h_right(i,j) = abs(xx_right(i,j) * yy_right(i,j) - 2 * xy_right(i,j));
        end
    end
end

h_left_s = zeros(size(h_left));
for i = 2:size(left, 1)-1
    for j = 2:size(left, 2)-1
        if h_left(i,j) > h_left(i-1,j-1) && ...
                h_left(i,j) > h_left(i-1,j) && ...
                h_left(i,j) > h_left(i-1,j+1) && ...
                h_left(i,j) > h_left(i,j-1) && ...
                h_left(i,j) > h_left(i,j+1) && ...
                h_left(i,j) > h_left(i+1,j-1) && ...
                h_left(i,j) > h_left(i+1,j) && ...
                h_left(i,j) > h_left(i+1,j+1)
            h_left_s(i,j) = h_left(i,j);
        end
    end
end

h_right_s = zeros(size(h_right));
for i = 2:size(right, 1)-1
    for j = 2:size(right, 2)-1
        if h_right(i,j) > h_right(i-1,j-1) && ...
                h_right(i,j) > h_right(i-1,j) && ...
                h_right(i,j) > h_right(i-1,j+1) && ...
                h_right(i,j) > h_right(i,j-1) && ...
                h_right(i,j) > h_right(i,j+1) && ...
                h_right(i,j) > h_right(i+1,j-1) && ...
                h_right(i,j) > h_right(i+1,j) && ...
                h_right(i,j) > h_right(i+1,j+1)
            h_right_s(i,j) = h_right(i,j);
        end
    end
end

[sort_left, sort_left_i] = sort(h_left_s(:), 'descend');
[sort_right, sort_right_i] = sort(h_right_s(:), 'descend');

figure();
imshow([h_left h_right]);
hold on;

proj = zeros(2 * MAX_FEATURES, 1);
M = zeros(2 * MAX_FEATURES, 6);
for i = 1:MAX_FEATURES
    left_index = sort_left_i(i);
    x_left = floor(left_index / size(left, 1));
    y_left = mod(left_index, size(left, 1));
    x_closest = 0;
    y_closest = 0;
    distance = inf;
    for j = 1:MAX_FEATURES
        right_index = sort_right_i(j);
        x_right = floor(right_index / size(right, 1));
        y_right = mod(right_index, size(right, 1));
        temp_distance = sqrt((x_left - x_right)^2 + (y_left - y_right)^2);
        if temp_distance < distance
            x_closest = x_right;
            y_closest = y_right;
            distance = temp_distance;
        end
    end
    plot(x_left, y_left, 'r.');
    plot(x_closest + size(right, 2), y_closest, 'r.');
    line([x_left x_closest + size(right, 2)], [y_left y_closest]);
    
    proj((i-1)*2+1) = x_closest;
    proj((i-1)*2+2) = y_closest;
    M((i-1)*2+1,:) = [x_left y_left 1 0 0 0];
    M((i-1)*2+2,:) = [0 0 0 x_left y_left 1];
end

hold off;

transform = [reshape(pinv(M) * proj, 3, 2)'; 0 0 1];
t_left = imwarp(left, affine2d(transform'));
l_t_left = t_left - imfilter(imgaussfilt(t_left, s), ones(5,5)/25);
l_right = right - imfilter(imgaussfilt(right, s), ones(5,5)/25);

fused = zeros(size(right));

for i = 1:min(size(t_left,1),size(right,1))
    for j = 1:min(size(t_left,2),size(right,2))
        %fused(i,j) = l_t_left(i,j)/(l_t_left(i,j) + l_right(i,j)) * t_left(i,j) + ...
        %    l_right(i,j)/(l_t_left(i,j) + l_right(i,j)) * original_right(i,j);
        fused(i,j) = 0.5 * t_left(i,j) + 0.5 * right(i,j);
    end
end

figure();
imshow(fused); 

%figure();
%h = surf(1:size(h_left, 2), 1:size(h_left, 1), h_left, original_left);
%h.EdgeColor = 'none';
%xlim([0 1520]);
%ylim([0 2688]);