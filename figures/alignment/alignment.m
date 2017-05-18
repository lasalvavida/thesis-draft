original_left = imread('waterboy_left.png');
original_right = imread('waterboy_right.png');

left = im2double(rgb2gray(original_left));
right = im2double(rgb2gray(original_right));

m = 5;
s = 0.5;

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
ITERATIONS = 20;

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

pts_left = zeros(MAX_FEATURES, 2);
pts_right = zeros(MAX_FEATURES, 2);
for i = 1:MAX_FEATURES
    left_index = sort_left_i(i);
    right_index = sort_right_i(i);
    x_left = floor(left_index / size(left, 1));
    y_left = mod(left_index, size(left, 1));
    x_right = floor(right_index / size(right, 1));
    y_right = mod(right_index, size(right, 1));
    pts_left(i,:) = [x_left y_left];
    pts_right(i,:) = [x_right y_right];
end

transform = eye(3);
num_features = MAX_FEATURES;
valid = ones(MAX_FEATURES,1);
distances = zeros(MAX_FEATURES,1);
mappings = zeros(MAX_FEATURES,1);
for i = 1:ITERATIONS
    proj = zeros(2 * num_features, 1);
    M = zeros(2 * num_features, 6);
    errors = zeros(MAX_FEATURES,1);
    p = 1;
    for j = 1:MAX_FEATURES
        if valid(j) == 1
            x_left = pts_left(j,1);
            y_left = pts_left(j,2);
            distance = inf;
            for k = 1:MAX_FEATURES
                x_right = pts_right(k,1);
                y_right = pts_right(k,2);
                temp_distance = sqrt((x_left - x_right)^2 + (y_left - y_right)^2);
                if temp_distance < distance
                    x_closest = x_right;
                    y_closest = y_right;
                    mappings(j) = k;
                    distance = temp_distance;
                end
            end
            distances(j) = distance;
            proj((p-1)*2+1) = x_closest;
            proj((p-1)*2+2) = y_closest;
            M((p-1)*2+1,:) = [x_left y_left 1 0 0 0];
            M((p-1)*2+2,:) = [0 0 0 x_left y_left 1];
            p = p + 1;
        end
    end
    compute = [reshape(pinv(M) * proj, 3, 2)'; 0 0 1];
    for j = 1:MAX_FEATURES
        if valid(j) == 1
            pt_left = [pts_left(j,:) 1];
            pt_left = compute * pt_left';
            x_closest = pts_right(mappings(j),1);
            y_closest = pts_right(mappings(j),2);
            temp_distance = sqrt((pt_left(1) - x_closest)^2 + (pt_left(2) - y_closest)^2);
            errors(j) = abs(abs(distances(j)) - abs(temp_distance));
            pts_left(j,:) = pt_left(1:2);
        end
    end
    [sort_error, sort_error_i] = sort(errors, 'descend');
    for j = 1:min(length(errors),num_features*0.05)
        valid(sort_error_i(j)) = 0;
    end
    transform = compute * transform;
end

t_left = imwarp(original_left, affine2d(transform'), 'OutputView', imref2d(size(original_left)));

fused = imfuse(t_left, original_right);

figure();
imshow(fused);
hold on;
for i = 1:MAX_FEATURES
    if valid(i) == 1
        plot(pts_left(i,1), pts_left(i,2), 'r.');
    end
    plot(pts_right(i,1), pts_right(i,2), 'b.');
end
hold off;

%figure();
%h = surf(1:size(h_left, 2), 1:size(h_left, 1), h_left, original_left);
%h.EdgeColor = 'none';
%xlim([0 1520]);
%ylim([0 2688]);