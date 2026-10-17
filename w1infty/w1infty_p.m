close all;clear;clc
% 读取数据
w1infty_1 = load('w1infty_1000_16.txt');
energy_1 = load('energy_1000_16.txt');
w1infty_2 = load('w1infty_1000_32.txt');
energy_2 = load('energy_1000_32.txt');

figure;
plot(w1infty_1(:,1), w1infty_1(:,2)); hold on
plot(energy_1(:,1), energy_1(:,2)); hold on
plot(w1infty_2(:,1), w1infty_2(:,2)); hold on
plot(energy_2(:,1), energy_2(:,2));

legend('$\Vert\nabla\mathbf{m}_h\Vert_{L^\infty}$\  for\  $\widehat h=\frac{1}{8}$',...
    '$\frac{1}{2}\Vert\nabla\mathbf{m}_h\Vert_h^2$\  for\  $\widehat h=\frac{1}{8}$',...
    '$\Vert\nabla\mathbf{m}_h\Vert_{L^\infty}$\  for\  $\widehat h=\frac{1}{16}$',... 
    '$\frac{1}{2}\Vert\nabla\mathbf{m}_h\Vert_h^2$\  for\  $\widehat h=\frac{1}{16}$',"Interpreter", "latex")
xlabel("$\mathbf{t}$", "Interpreter", "latex")
% ylabel("$\frac{1}{2}\Vert\mathbf{m}_h\Vert^2$", "Interpreter", "latex")
