%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Matlab code to analyse the performance of image quality metrics on the 
%Colourlab Image Database: Image Quality. If you are using the database 
%and this code please cite our work. 
%Written by Marius Pedersen (marius.pedersen@hig.no)
% Version 1.0.2 - 31.10.14
%22.09.14: added some additional plots, calculation of average
%correlation values and correlation for each distortion. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Citation:
%Xinwei Liu, Marius Pedersen and Jon Yngve Hardeberg. CID:IQ - A New Image 
%Quality Database. International Conference on Image and Signal Processing 
%(ICISP 2014). Vol. 8509. Cherbourg, France. Jul., 2014.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = CorrCIDIQ()
clear all; %clear any variables that might be in the workspace
close all; %close all figures
d=100; 
uiopen(); %displays window to open the result file
if(d == 50) %if the distance is 50 cm the load subjective scores for this distance
    load MOS50.mat
    MOS = MOS50; 
    clear MOS50; 
elseif(d ==100) %if the distance is 100 cm the load subjective scores for this distance
    load MOS100.mat
    MOS = MOS100; 
    clear MOS100; 
else
    error('Unknown distance')
end
names = fieldnames(Results);
NumberDatapoints = 690; %the entire database contains 690 images, which is important for the calculation of the confidence intervals
warning off; 


%% Parameters for subplot
TotalNumberofMetrics = length(names)-2; 
if(TotalNumberofMetrics == 1) %if only one metric has been calculated
    X_NumberofMetrics = 1; 
    Y_NumberofMetrics = 1;
else
    if(TotalNumberofMetrics == 2) %if two metrics have been calculated
        X_NumberofMetrics = TotalNumberofMetrics;
    else(floor(TotalNumberofMetrics/2) == 1) %if more than 2
    X_NumberofMetrics = ceil(sqrt(TotalNumberofMetrics));      
    end
    if(TotalNumberofMetrics - X_NumberofMetrics == 0)
        Y_NumberofMetrics = 1; 
    else
        Y_NumberofMetrics = ceil(TotalNumberofMetrics./X_NumberofMetrics);
    end
end

CounterMetrics = 0;

figure('units','normalized','outerposition',[0 0 1 1]);

%% Loop through all metrics
    for k=1:1:length(names) %loop through all metrics
    if(strcmp(names(k), 'Original_Name')) %skip since it is not a metric
    elseif(strcmp(names(k), 'Reproduction_Name')) %skip since it is not a metric        
    else
        CounterMetrics = CounterMetrics +1; %increasing counter for metrics
        disp(['Calculating scores for metric #', num2str(CounterMetrics),': ', char(names(k))]); 
        ResultsCID = getfield(Results,char(names(k))); %extracting results for a single metric 
        XtickNames(CounterMetrics) = names(k);
        
        %Making a plot for the metric
        subplot(X_NumberofMetrics,Y_NumberofMetrics,CounterMetrics); %subplot used to plot all metrics in the same window
        p = plot(ResultsCID,MOS,'+');
        set(p,'Color','blue','LineWidth',1);
        
        %Linear correlation for all images%
        
        %If the results for a metric are 'inf'/'nan' then put the maximum to a value that is not 'inf'/'nan', in this case the maximum value that is not 'inf'
        ResultsCID(isnan(ResultsCID))=0; %If nan's then set these to 0
        ResultsCID(isinf(ResultsCID))=max(ResultsCID(isfinite(ResultsCID))); %If Inf set value to the maximum value 
       
        Linearcorr_coefPearson(CounterMetrics) = corr(MOS, ResultsCID', 'type','Pearson'); %pearson linear coefficient
        Linearcorr_coefSpearman(CounterMetrics) = corr(MOS, ResultsCID', 'type','Spearman'); %pearson linear coefficient
        Linearcorr_coefKendall(CounterMetrics) = corr(MOS, ResultsCID', 'type','Kendall'); %pearson linear coefficient

        LinearHigh_limit_coefPearson(CounterMetrics) = ((exp(2*(1/2*log((1+Linearcorr_coefPearson(CounterMetrics))/(1-Linearcorr_coefPearson(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+Linearcorr_coefPearson(CounterMetrics))/(1-Linearcorr_coefPearson(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))+1));
        LinearLow_limit_coefPearson(CounterMetrics) = (exp(2*(1/2*log((1+Linearcorr_coefPearson(CounterMetrics))/(1-Linearcorr_coefPearson(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+Linearcorr_coefPearson(CounterMetrics))/(1-Linearcorr_coefPearson(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))+1);

        LinearHigh_limit_coefSpearman(CounterMetrics) = ((exp(2*(1/2*log((1+Linearcorr_coefSpearman(CounterMetrics))/(1-Linearcorr_coefSpearman(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+Linearcorr_coefSpearman(CounterMetrics))/(1-Linearcorr_coefSpearman(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))+1));
        LinearLow_limit_coefSpearman(CounterMetrics) = (exp(2*(1/2*log((1+Linearcorr_coefSpearman(CounterMetrics))/(1-Linearcorr_coefSpearman(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+Linearcorr_coefSpearman(CounterMetrics))/(1-Linearcorr_coefSpearman(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))+1);

        
         %Non-linear correlation for all images%
        
        [mosFit , ypreFit,bayta ehat,J ] = LogisticFitting(MOS,ResultsCID');
        RMSE(CounterMetrics) = sqrt(sum((ypreFit - mosFit).^2) / length(mosFit));%root meas squared error
        corr_coefPearson(CounterMetrics) = corr(mosFit, ypreFit, 'type','Pearson'); %pearson linear coefficient
        corr_coefSpearman(CounterMetrics) = corr(mosFit, ypreFit, 'type','Spearman'); %pearson linear coefficient
        corr_coefKendall(CounterMetrics) = corr(mosFit, ypreFit, 'type','Kendall'); %pearson linear coefficient

        
        %calculating 95% confidence interval for non-linear correlation
        High_limit_coefPearson(CounterMetrics) = ((exp(2*(1/2*log((1+corr_coefPearson(CounterMetrics))/(1-corr_coefPearson(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+corr_coefPearson(CounterMetrics))/(1-corr_coefPearson(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))+1));
        Low_limit_coefPearson(CounterMetrics) = (exp(2*(1/2*log((1+corr_coefPearson(CounterMetrics))/(1-corr_coefPearson(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+corr_coefPearson(CounterMetrics))/(1-corr_coefPearson(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))+1);

        High_limit_coefSpearman(CounterMetrics) = ((exp(2*(1/2*log((1+corr_coefSpearman(CounterMetrics))/(1-corr_coefSpearman(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+corr_coefSpearman(CounterMetrics))/(1-corr_coefSpearman(CounterMetrics))) + 1.96*(1/(sqrt(NumberDatapoints-3)))))+1));
        Low_limit_coefSpearman(CounterMetrics) = (exp(2*(1/2*log((1+corr_coefSpearman(CounterMetrics))/(1-corr_coefSpearman(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))-1)/(exp(2*(1/2*log((1+corr_coefSpearman(CounterMetrics))/(1-corr_coefSpearman(CounterMetrics))) - 1.96*(1/(sqrt(NumberDatapoints-3)))))+1);

      
        %adding the non-linear regression line to the plot
        t = min(ResultsCID):0.01:max(ResultsCID(isfinite(ResultsCID)));
        [ypreFit junk] = nlpredci(@logistic,t,bayta,ehat,J); %Nonlinear regression prediction confidence intervals 
        hold on;
        p = plot(t,ypreFit);%plotting the nonlinear regression line
        set(p,'Color','black','LineWidth',2); %setting it to have a wider line width and black
       % legend('Images in CID:IQ','Curve fitted with logistic function', 'Location','NorthWest'); %adding legend
       % xlabel('Objective score by metric'); %adding xlabel 
       % ylabel(['MOS (', num2str(d), ' cm)']); % adding ylabel 
        title(char(names(k))); %adding title, being the name of the metric

        %Calculating the correlation per image
        counterPerImage = 0; 
        for IP=1:5:690
            counterPerImage = counterPerImage +1;
            pearsonCorrIP(counterPerImage) = corr(ResultsCID(IP:IP+4)',MOS(IP:IP+4),'type','pearson');
            spearmanCorrIP(counterPerImage) = corr(ResultsCID(IP:IP+4)',MOS(IP:IP+4),'type','spearman');
            kendallCorrIP(counterPerImage) = corr(ResultsCID(IP:IP+4)',MOS(IP:IP+4),'type','kendall');
        end

        AveragePearsonCorrelationPerImage(CounterMetrics) = mean(pearsonCorrIP);
        AverageSpearmanCorrelationPerImage(CounterMetrics) = mean(spearmanCorrIP);
        AverageKendallCorrelationPerImage(CounterMetrics) = mean(kendallCorrIP);
        

        %Calculate results per distortion
        clear  mosFit ypreFit bayta ehat J 
        DistortionNumber = 0;
        for IP=1:115:690     
            DistortionNumber = DistortionNumber +1;
            NumberDatapointsDistortion = 115;
            [mosFit , ypreFit,bayta ehat,J ] = LogisticFitting(MOS(IP:IP+114),ResultsCID(IP:IP+114)');
            RMSE_distortion(CounterMetrics,DistortionNumber) = sqrt(sum((ypreFit - mosFit).^2) / length(mosFit));%root meas squared error
            corr_coefPearson_distortion(CounterMetrics,DistortionNumber) = corr(mosFit, ypreFit, 'type','Pearson'); %pearson linear coefficient
            corr_coefSpearman_distortion(CounterMetrics,DistortionNumber) = corr(mosFit, ypreFit, 'type','Spearman'); %pearson linear coefficient
            corr_coefKendall_distortion(CounterMetrics,DistortionNumber) = corr(mosFit, ypreFit, 'type','Kendall'); %pearson linear coefficient


            %calculating 95% confidence interval for non-linear correlation
            High_limit_coefPearson_distortion(CounterMetrics,DistortionNumber) = ((exp(2*(1/2*log((1+corr_coefPearson_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefPearson_distortion(CounterMetrics,DistortionNumber))) + 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))-1)/(exp(2*(1/2*log((1+corr_coefPearson_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefPearson_distortion(CounterMetrics,DistortionNumber))) + 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))+1));
            Low_limit_coefPearson_distortion(CounterMetrics,DistortionNumber) = (exp(2*(1/2*log((1+corr_coefPearson_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefPearson_distortion(CounterMetrics,DistortionNumber))) - 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))-1)/(exp(2*(1/2*log((1+corr_coefPearson_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefPearson_distortion(CounterMetrics,DistortionNumber))) - 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))+1);

            High_limit_coefSpearman_distortion(CounterMetrics,DistortionNumber) = ((exp(2*(1/2*log((1+corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))) + 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))-1)/(exp(2*(1/2*log((1+corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))) + 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))+1));
            Low_limit_coefSpearman_distortion(CounterMetrics,DistortionNumber) = (exp(2*(1/2*log((1+corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))) - 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))-1)/(exp(2*(1/2*log((1+corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))/(1-corr_coefSpearman_distortion(CounterMetrics,DistortionNumber))) - 1.96*(1/(sqrt(NumberDatapointsDistortion-3)))))+1);

        end 
        
        
        
        %Clearing variables
        clear  mosFit ypreFit bayta ehat J  Maximum ResultsCID p t junk DistortionNumber 
        
    end
end

%% Barplot Pearson correlation with confidence intervals%% 

High_limit_coefPearson = High_limit_coefPearson -corr_coefPearson; %finding the value to add to the bar plot for the confidence interval
Low_limit_coefPearson = corr_coefPearson- Low_limit_coefPearson;  %finding the value to add to the bar plot for the confidence interval

figure; 
barSettings = {};
lineSettings = {High_limit_coefPearson', 'linestyle', 'none','linewidth',2}; 
x = 1:CounterMetrics; % x-axis 
errorbarbar(x, corr_coefPearson', Low_limit_coefPearson', barSettings, lineSettings);
axis([0.5 CounterMetrics+0.5 0 1]) %setting the axis to go from 0 to 1 for the correlation
ylabel('Pearson Correlation')
xlabel('Metrics')
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(strcat('Pearson correlation values with a 95% confidence interval. Viewing distance ', num2str(d), ' cm'))


%% Barplot Spearman correlation with confidence intervals
High_limit_coefSpearman = High_limit_coefSpearman -corr_coefSpearman; %finding the value to add to the bar plot for the confidence interval
Low_limit_coefSpearman = corr_coefSpearman- Low_limit_coefSpearman;  %finding the value to add to the bar plot for the confidence interval

figure; 
barSettings = {};
lineSettings = {High_limit_coefSpearman', 'linestyle', 'none','linewidth',3}; 
x = 1:CounterMetrics; % x-axis 
errorbarbar(x, corr_coefSpearman', Low_limit_coefSpearman', barSettings, lineSettings);
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Spearman correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Spearman correlation values with a 95% confidence interval. Viewing distance ', num2str(d), ' cm']);

%% Barplot Kendall correlation for each image
figure; 
bar(corr_coefKendall); 
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Kendall correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Kendall correlation. Viewing distance ', num2str(d), ' cm']);

%% barplot RMSE 
figure; 
bar(RMSE);
ylabel('RMSE');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['RMSE. Viewing distance ', num2str(d), ' cm']);

%% Barplot average Pearson correlation for each image% Please note that this is the correlation coefficient for one image for one distortion (i.e. for each original given one distortion the correlation is taken, then averaged over the all values)
figure; 
%Image difference metrics will have an opposite correlation compared
%to other metrics, and therefore their correlation needs to be multiplied
%with -1 in order to compare them to the other metrics. This is here done
%simply by multiplying negative values with -1 (please note that some
%metrics might have a negative value without being image difference
%metrics)
AveragePearsonCorrelationPerImage(AveragePearsonCorrelationPerImage<0) = AveragePearsonCorrelationPerImage(AveragePearsonCorrelationPerImage<0)*-1; 
bar(AveragePearsonCorrelationPerImage); 
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Average Pearson correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Average Pearson correlation per image. Viewing distance ', num2str(d), ' cm']);

%% Barplot average Pearson correlation for each image% Please note that this is the correlation coefficient for one image for one distortion (i.e. for each original given one distortion the correlation is taken, then averaged over the all values)
figure; 
%Image difference metrics will have an opposite correlation compared
%to other metrics, and therefore their correlation needs to be multiplied
%with -1 in order to compare them to the other metrics. This is here done
%simply by multiplying negative values with -1 (please note that some
%metrics might have a negative value without being image difference
%metrics)
AverageSpearmanCorrelationPerImage(AverageSpearmanCorrelationPerImage<0) = AverageSpearmanCorrelationPerImage(AverageSpearmanCorrelationPerImage<0)*-1; 
bar(AverageSpearmanCorrelationPerImage); 
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Average Spearman correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Average Spearman correlation per image. Viewing distance ', num2str(d), ' cm']);

%% Barplot average Kendall correlation for each image% Please note that this is the correlation coefficient for one image for one distortion (i.e. for each original given one distortion the correlation is taken, then averaged over the all values)
figure; 
%Image difference metrics will have an opposite correlation compared
%to other metrics, and therefore their correlation needs to be multiplied
%with -1 in order to compare them to the other metrics. This is here done
%simply by multiplying negative values with -1 (please note that some
%metrics might have a negative value without being image difference
%metrics)
AverageKendallCorrelationPerImage(AverageKendallCorrelationPerImage<0) = AverageKendallCorrelationPerImage(AverageKendallCorrelationPerImage<0)*-1; 
bar(AverageKendallCorrelationPerImage); 
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Average Kendall correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Average Kendall correlation per image. Viewing distance ', num2str(d), ' cm']);


%% Barplot Pearson correlation with confidence intervals for each distortion%% 
TextDist = {' JPEG2000','JPEG',' Poisson Noise',' Gaussian Blur',' SGCK gamut mapping',' \Delta E_{ab} gamut mapping'};
 for i=1:6
    High_limit_coefPearson = High_limit_coefPearson_distortion(:,i) -corr_coefPearson_distortion(:,i); %finding the value to add to the bar plot for the confidence interval
    Low_limit_coefPearson = corr_coefPearson_distortion(:,i)- Low_limit_coefPearson_distortion(:,i);  %finding the value to add to the bar plot for the confidence interval

    figure; 
    barSettings = {};
    lineSettings = {High_limit_coefPearson', 'linestyle', 'none','linewidth',2}; 
    x = 1:CounterMetrics; % x-axis 
    errorbarbar(x, corr_coefPearson_distortion(:,i)', Low_limit_coefPearson', barSettings, lineSettings);
    axis([0.5 CounterMetrics+0.5 0 1]) %setting the axis to go from 0 to 1 for the correlation
    ylabel('Pearson Correlation')
    xlabel('Metrics')
    set(gca,'XTickLabel',XtickNames);
    set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
    xticklabel_rotate([],90,[],'Fontsize',12);
    title(strcat('Pearson correlation values with a 95% confidence interval for distortion ',TextDist{i} ,'. Viewing distance: ', num2str(d), ' cm'))
 end


 %% Barplot linear Pearson correlation with confidence intervals%% 

LinearHigh_limit_coefPearson = LinearHigh_limit_coefPearson -Linearcorr_coefPearson; %finding the value to add to the bar plot for the confidence interval
LinearLow_limit_coefPearson = Linearcorr_coefPearson- LinearLow_limit_coefPearson;  %finding the value to add to the bar plot for the confidence interval

figure; 
barSettings = {};
lineSettings = {LinearHigh_limit_coefPearson', 'linestyle', 'none','linewidth',2}; 
x = 1:CounterMetrics; % x-axis 
errorbarbar(x, Linearcorr_coefPearson', LinearLow_limit_coefPearson', barSettings, lineSettings);
axis([0.5 CounterMetrics+0.5 -1 1]) %setting the axis to go from 0 to 1 for the correlation
ylabel('Pearson Correlation')
xlabel('Metrics')
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(strcat('Linear Pearson correlation values with a 95% confidence interval. Viewing distance ', num2str(d), ' cm'))

%% Barplot Linear Spearman correlation with confidence intervals
LinearHigh_limit_coefSpearman = LinearHigh_limit_coefSpearman -Linearcorr_coefSpearman; %finding the value to add to the bar plot for the confidence interval
LinearLow_limit_coefSpearman = Linearcorr_coefSpearman- LinearLow_limit_coefSpearman;  %finding the value to add to the bar plot for the confidence interval

figure; 
barSettings = {};
lineSettings = {LinearHigh_limit_coefSpearman', 'linestyle', 'none','linewidth',3}; 
x = 1:CounterMetrics; % x-axis 
errorbarbar(x, Linearcorr_coefSpearman', LinearLow_limit_coefSpearman', barSettings, lineSettings);
axis([0.5 CounterMetrics+0.5 0 1]); %setting the axis to go from 0 to 1 for the correlation
ylabel('Spearman correlation');
xlabel('Metrics');
set(gca,'XTickLabel',XtickNames);
set(gca,'XTick',[1.1:1:CounterMetrics+0.1]);
xticklabel_rotate([],90,[],'Fontsize',12);
title(['Linear Spearman correlation values with a 95% confidence interval. Viewing distance ', num2str(d), ' cm']);

 
%% Export to Latex
%Exporting all results to a latex table. This can be used in any latex
%document by using  /input{PearsonCorrelationNonLinear.tex}
columnLabels = {'Pearson', 'Spearman', 'Kendall'};
rowLabels = XtickNames; 
matrix2latex(corr_coefPearson', ['PearsonCorrelationNonLinear_', num2str(d) ,'cm.tex'], 'columnLabels', {'Pearson'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Pearson correlation values
matrix2latex(corr_coefSpearman', ['SpearmanCorrelationNonLinear_', num2str(d),'cm.tex'], 'columnLabels', {'Spearman'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Spearman correlation values
matrix2latex(corr_coefKendall', ['KendallCorrelationNonLinear_',num2str(d),'cm.tex'], 'columnLabels', {'Kendall'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Kendall correlation values

matrix2latex([corr_coefPearson; corr_coefSpearman; corr_coefKendall;]', ['CorrelationNonLinear_',num2str(d),'cm.tex'], 'columnLabels', columnLabels, 'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Pearson, Spearman and Kendall correlation values

matrix2latex(RMSE', ['RMSE_',num2str(d),'cm.tex'], 'columnLabels', {'RMSE'}, 'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with RMSE values


%For Average correlation per image%
matrix2latex(AveragePearsonCorrelationPerImage', ['PearsonCorrelationAveragePerImage_', num2str(d) ,'cm.tex'], 'columnLabels', {'Pearson'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Pearson correlation values
matrix2latex(AverageSpearmanCorrelationPerImage', ['SpearmanCorrelationAveragePerImage_', num2str(d),'cm.tex'], 'columnLabels', {'Spearman'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Spearman correlation values
matrix2latex(AverageKendallCorrelationPerImage', ['KendallCorrelationAveragePerImage_',num2str(d),'cm.tex'], 'columnLabels', {'Kendall'},'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Kendall correlation values

matrix2latex([AveragePearsonCorrelationPerImage; AverageSpearmanCorrelationPerImage; AverageKendallCorrelationPerImage;]', ['CorrelationAveragePerImage_',num2str(d),'cm.tex'], 'columnLabels', columnLabels, 'rowLabels', rowLabels, 'alignment', 'c', 'format', '%-6.3f'); %table with Pearson, Spearman and Kendall correlation values

end