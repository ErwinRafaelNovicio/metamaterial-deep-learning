clc
clear all
close all


%% laod file


DATA = load("Parameter_sweep.txt");

% prima colonna --> campo magnetico
% seconda colonna --> TPU_L
% terza colonna --> TPU_R
% quarta colonna --> time
% quinta colonna --> stress
% sesta colonna --> strain
% settima --> TPU_thickness
% ottava ---> MR_l length
% nona --> MR fluid radius
% decima --> displacement


%% BLOCK extraction for B


B_values = [0 0.5 1 1.5 2];

Bmat = cell(length(B_values),1);

for i = 1:length(B_values)
    val = B_values(i);
    Bmat{i} = DATA(DATA(:, 1) == val, :);
end

for kkk = 1:length(Bmat)
figure()
ex = Bmat{kkk};

%% BLOCK EXTRACTION for TPU_L

% now with B fixed and based on TPU_L

TPU_L_values = [20:5:50];

TPU_L_mat = cell(length(TPU_L_values),1);

for ii = 1:length(TPU_L_values)
        val2 = TPU_L_values(ii);
        TPU_L_mat{ii} = ex(ex(:, 2) == val2, :);
end
for kk = 1:length(TPU_L_mat)
    ex1 = TPU_L_mat{kk};
%% BLOCK EXTRACTION for TPU_R


% now with B fixed and based on TPU_L

TPU_R_values = double([3:3:9]);

TPU_R_mat = cell(length(TPU_R_values),1);

for iii = 1:length(TPU_R_values)
        val3 = TPU_R_values(iii);
        TPU_R_mat{iii} = ex1(round(ex1(:, 3)) == val3, :);
end
%% plotting
subplot(length(TPU_L_mat),2,kk)
    for k = 1:length(TPU_R_mat)
        ex2 =  TPU_R_mat{k};

        
        hold on
        plot(ex2(:,6),ex2(:,5))
        legend()
        xlabel('Strain [1]')
        ylabel('Stress [Pa]')
        end
    end
end
