close all;clear;clc
% 读取数据
energy_1 = load('energy_10.txt');
energy_2 = load('energy_20.txt');
energy_3 = load('energy_40.txt');
energy_4 = load('energy_80.txt');
energy_5 = load('energy_160.txt');


% w1infty_1 = load('w1infty_50_2.txt');
% w1infty_2 = load('w1infty_100_2.txt');
% w1infty_3 = load('w1infty_200_2.txt');
% w1infty_4 = load('w1infty_500_2.txt');
% w1infty_5 = load('w1infty_1000_2.txt');


% 绘制
figure;
plot(energy_1(:,1), energy_1(:,2)); hold on
plot(energy_2(:,1), energy_2(:,2)); hold on
plot(energy_3(:,1), energy_3(:,2)); hold on
plot(energy_4(:,1), energy_4(:,2)); hold on
plot(energy_5(:,1), energy_5(:,2)); 
legend('$\Delta t=1$', ...
    '$\Delta t=0.5$', ...
    '$\Delta t=0.25$', ...
    '$\Delta t=0.125$', ...
    '$\Delta t=0.0625$', "Interpreter", "latex")
xlabel("$\mathbf{t}$", "Interpreter", "latex")
ylabel("$\frac{1}{2}\Vert\nabla\mathbf{m}_h\Vert_h^2$", "Interpreter", "latex")
