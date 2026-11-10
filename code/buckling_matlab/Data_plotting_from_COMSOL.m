clear; clc; close all;

% =========================================================================
% SECTION: USER SELECTION
% =========================================================================
% 1 = Stress-Strain only
% 2 = Force-Displacement only
% 3 = Energy Balance (Kinetic vs Stored) only
% 20 = Run all cases
plotSelection = 1; 

% =========================================================================
% SECTION: FILE PATHS & DIRECTORY
% =========================================================================
baseDir = 'C:\Users\AlexM\OneDrive - Politecnico di Milano\Documenti\Università Alex\Polimi\PhD\Ricerca\Metamaterials\COMSOL_models\Axial_simmetric\TPU\Results'; 

% Specific filenames as exported from COMSOL
fileSS = 'Stress_strain.txt';
fileFD = 'Force_disp.txt';
fileKE = 'Kinetic.txt';

% =========================================================================
% SECTION: PLOT PARAMETERS (Visual Settings)
% =========================================================================
params.LineWidth = 1.5;
params.FontSize  = 14;
params.FontWeight = 'bold';
params.Grid      = 'on';
params.Marker    = 'none';
params.AxisMargin = 0.05;
params.FigWidth_cm  = 32;         
params.FigHeight_cm = 18;
params.ColorSS   = [0 0.447 0.741];      % Blue
params.ColorFD   = [0 0.447 0.741];      % Blue
params.ColorKin  = [0.929 0.694 0.125];  % Yellow (Kinetic)
params.ColorSto  = [0.466 0.674 0.188];  % Green (Stored)

% =========================================================================
% SECTION: PLOTTING LOGIC (Switch-Case)
% =========================================================================

% Determine which cases to execute
if plotSelection == 20
    tasks = 1:3;
else
    tasks = plotSelection;
end

for t = tasks
    switch t
        case 1
            %% CASE 1: STRESS-STRAIN 
            data = readmatrix(fullfile(baseDir, fileSS), 'FileType', 'text', 'CommentStyle', '%');
            assignin('base', 'stress_strain', data); 
            
            x = data(:,1); y = data(:,2); 
            createFigure('Stress-Strain Analysis', params);
            plot(x, y, 'LineWidth', params.LineWidth, 'Color', params.ColorSS);
            
            xlabel('Strain [-]', 'FontWeight', params.FontWeight);
            ylabel('Stress [Pa]', 'FontWeight', params.FontWeight);
            title('Engineering Stress-Strain Curve', 'FontSize', params.FontSize+2);
            setSmartLimits(x, y, params.AxisMargin);
            applyStyle(params);

        case 2
            %% CASE 2: FORCE-DISPLACEMENT [cite: 1, 2]
            data = readmatrix(fullfile(baseDir, fileFD), 'FileType', 'text', 'CommentStyle', '%');
            assignin('base', 'force_disp', data); 
            
            x = data(:,1); y = data(:,2); 
            createFigure('Force-Displacement Analysis', params);
            plot(x, y, 'LineWidth', params.LineWidth, 'Color', params.ColorFD);
            
            xlabel('Displacement [mm]', 'FontWeight', params.FontWeight);
            ylabel('Force [N]', 'FontWeight', params.FontWeight);
            title('Force vs. Displacement', 'FontSize', params.FontSize+2);
            setSmartLimits(x, y, params.AxisMargin);
            applyStyle(params);

        case 3
            %% CASE 3: ENERGY BALANCE [cite: 11, 12, 13]
            data = readmatrix(fullfile(baseDir, fileKE), 'FileType', 'text', 'CommentStyle', '%');
            assignin('base', 'energy_data', data); 
            
            time = data(:,1); kin = data(:,2); sto = data(:,3); 
            createFigure('Energy Balance Analysis', params);
            hold on;
            plot(time, sto, 'LineWidth', params.LineWidth, 'Color', params.ColorSto, 'DisplayName', 'Elastic Energy');
            plot(time, kin, 'LineWidth', params.LineWidth, 'Color', params.ColorKin, 'DisplayName', 'Kinetic Energy');
            
            xlabel('Time [s]', 'FontWeight', params.FontWeight);
            ylabel('Energy [J]', 'FontWeight', params.FontWeight);
            title('Energy Comparison (Inertial Check)', 'FontSize', params.FontSize+2);
            legend('Location', 'best', 'FontSize', params.FontSize-2);
            setSmartLimits(time, [kin; sto], params.AxisMargin);
            applyStyle(params);
            
        otherwise
            fprintf('Selection %d not recognized.\n', t);
    end
end

% =========================================================================
% UNIVERSAL HELPER FUNCTIONS
% =========================================================================

function createFigure(figName, p)
    % Creates a figure with a fixed 16:9 ratio in centimeters
    hFig = figure('Name', figName, 'Color', 'w', 'Units', 'centimeters');
    % Position: [left, bottom, width, height]
    set(hFig, 'Position', [5, 5, p.FigWidth_cm, p.FigHeight_cm]);
end

function setSmartLimits(x, y, marginPerc)
    % Padding is calculated as a percentage of the global MAX
    yMax = max(y, [], 'all', 'omitnan');
    yMin = min(y, [], 'all', 'omitnan');
    yPadding = yMax * marginPerc;
    
    xMax = max(x, [], 'all', 'omitnan');
    xMin = min(x, [], 'all', 'omitnan');
    xPadding = xMax * marginPerc;

    ylim([yMin - yPadding, yMax + yPadding]);
    xlim([xMin - xPadding, xMax + xPadding]);
end

function applyStyle(p)
    grid(p.Grid);
    set(gca, 'FontSize', p.FontSize, ...
             'FontWeight', p.FontWeight, ...
             'Box', 'on', ...
             'LineWidth', 1.5, ...
             'TickLabelInterpreter', 'tex');
end