function [tr_data] = ANNmapping_to_python(path,layers,epochLimit)
    myVars = {'Time','u1','du1','u2','du2',...
                'x1','dx1','d2x1',...
                'x3','dx3','d2x3',...
                'x5','dx5','d2x5',...
                'fT1','dfT1','d2fT1',...
                'fT2','dfT2','d2fT2'};
    all_data = load(path+"babblingTrial_outputData.mat",myVars{:});
    
    % Defining Input/Output data
    input_data = struct;
    input_data.all=[all_data.x3;all_data.dx3;all_data.d2x3;...
                    all_data.x5;all_data.dx5;all_data.d2x5;...
                    all_data.fT1;all_data.dfT1;all_data.d2fT1;...
                    all_data.fT2;all_data.dfT2;all_data.d2fT2];
    input_data.bio=[all_data.x3;all_data.dx3;...
                    all_data.x5;all_data.dx5;...
                    all_data.fT1;all_data.fT2];
    input_data.kinapprox=[all_data.x3;all_data.dx3;...
                            all_data.x5;all_data.dx5];
    input_data.allmotor=[all_data.x3;all_data.dx3;all_data.d2x3;...
                        all_data.x5;all_data.dx5;all_data.d2x5];

    output_data=all_data.x1;
    % plot(output_data)
    %% Neural Network Training/Testing
    tr_data = struct;
    fn = fieldnames(input_data);
    for i=1:numel(fn)
        % Defining Test/Train data
        In=input_data.(fn{i});
        Out=output_data;
        [trainInd,~,testInd] = dividerand(size(In,2),.9,0,.1); % 90% train and 10% test
        % NN - Train
        net=feedforwardnet(double(layers));
        net.trainParam.epochs = double(epochLimit);
        [net, tr] = train(net,In(:,trainInd),Out(trainInd));
        tr_data.(fn{i}) = struct;
        tr_data.(fn{i}).tr = tr;
%         tr_data.(fn{i}).net = net; ### CANT SAVE NETS TO PYTHON.
        % view(net)
        % NN - Test
        close all;
        tr_data.(fn{i}).predicted_out=net(In(:,testInd)); % using ANN to map test data
        tr_data.(fn{i}).test_error = sqrt(mean((Out(testInd)-tr_data.(fn{i}).predicted_out).^2));
    end
end
