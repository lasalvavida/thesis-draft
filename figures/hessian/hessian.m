d = 0.05;
x = -5:d:5;
y = -5:d:5;
s = 1;

Gxx = zeros(length(x), length(y));
Gxy = zeros(length(x), length(y));
Gyy = zeros(length(x), length(y));

for i = 1:length(x)
    for j = 1:length(y)
        x_i = x(i);
        y_j = y(j);
        
        Gxx(i,j) = (-1 + x_i^2 / s^2) * exp(-(x_i^2 + y_j^2)/(2 * s^2))/(2 * pi * s^4);
        Gxy(i,j) = (x_i * y_j) / (2 * pi * s^6) * exp(-(x_i^2 + y_j^2)/(2 * s^2));
        Gyy(i,j) = (-1 + y_j^2 / s^2) * exp(-(x_i^2 + y_j^2)/(2 * s^2))/(2 * pi * s^4);
    end
end

figure();
h = surf(x, y, Gxx);
h.EdgeColor = 'none';
xlabel('x_1');
ylabel('x_2');
zlabel('G');
grid on;

figure();
h = surf(x, y, Gxy);
h.EdgeColor = 'none';
xlabel('x_1');
ylabel('x_2');
zlabel('G');
grid on;

figure();
h = surf(x, y, Gyy);
h.EdgeColor = 'none';
xlabel('x_1');
ylabel('x_2');
zlabel('G');
grid on;