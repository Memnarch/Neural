unit Neural.Network;

interface

uses
  Neural.Neuron;

type
  TEpochProgression = reference to procedure(ACurrentEpoche: Integer);

  TNeuralNetwork = class
  private
    FOnEpochProgression: TEpochProgression;
  protected
    FNeurons: TArray<TArray<TNeuron>>;
    FLongestLayer: Integer;
    FInputCount: Integer;
    function GetNeuronClass: TNeuronClass; virtual; abstract;
    function InternalFeedForward(ANumInputs: Integer; ABuffer: TArray<TArray<Single>>): TArray<Single>;
  public
    destructor Destroy; override;
    procedure Build(const ANetwork: TArray<Integer>);
    function FeedForward(const AValues: TArray<Single>): TArray<Single>;
    procedure Train(const AData: TArray<TArray<Single>>; const AExpected: TArray<Single>);
    property OnEpochProgression: TEpochProgression read FOnEpochProgression write FOnEpochProgression;
  end;

  TNeuralNetwork<T: TNeuron> = class(TNeuralNetwork)
  protected
      function GetNeuronClass: TNeuronClass; override;
  end;

function MSELoss(const AExpected, ACalculated: TArray<Single>): Single;

implementation

function MSELoss(const AExpected, ACalculated: TArray<Single>): Single;
var
  i: Integer;
begin
  Result := 0;
  for i := Low(AExpected) to High(ACalculated) do
    Result := Result + Sqr(AExpected[i] - ACalculated[i]);
  Result := Result / Length(AExpected);
end;

function Derive(const Value: Single): Single;
begin
  Result := Value * (1 - Value);
end;

{ TNeuralNetwork }

procedure TNeuralNetwork.Build(const ANetwork: TArray<Integer>);
var
  i, k, m: Integer;
  LWeights: TArray<Single>;
begin
  FInputCount := ANetwork[0];
  FLongestLayer := FInputCount;
  SetLength(FNeurons, Length(ANetwork) - 1);
  for i := Low(FNeurons) to High(FNeurons) do //first line specifies Inputcount so start creating nodes at second
  begin
    SetLength(FNeurons[i], ANetwork[i + 1]);
    if ANetwork[i + 1] > FLongestLayer then
      FLongestLayer := ANetwork[i + 1];
    for k := 0 to Pred(ANetwork[i + 1]) do
    begin
      SetLength(LWeights, ANetwork[i]);
      for m := 0 to High(LWeights) do
        LWeights[m] := m;
      FNeurons[i, k] := GetNeuronClass().Create(0, LWeights);
      LWeights := nil;
    end;
  end;
end;

destructor TNeuralNetwork.Destroy;
var
  LNeuron: TNeuron;
  LNeurons: TArray<TNeuron>;
begin
  for LNeurons in FNeurons do
    for LNeuron in LNeurons do
      LNeuron.Free;
  inherited;
end;

function TNeuralNetwork.FeedForward(const AValues: TArray<Single>): TArray<Single>;
var
  LTemp: array [0..1] of TArray<Single>;
  LBuffer: TArray<TArray<Single>>;
  i: Integer;
begin
  SetLength(LTemp[0], FLongestLayer);
  SetLength(LTemp[1], FLongestLayer);
  for i := Low(AValues) to High(AValues) do
    LTemp[0][i] := AValues[i];
  SetLength(LBuffer, Length(FNeurons)+1);
  for i := Low(LBuffer) to High(LBuffer) do
    LBuffer[i] := LTemp[i mod 2];
  Result := InternalFeedForward(Length(AValues), LBuffer);
end;

function TNeuralNetwork.InternalFeedForward(ANumInputs: Integer; ABuffer: TArray<TArray<Single>>): TArray<Single>;
var
  LSource, LTarget, LSourceLength: Integer;
  i, k: Integer;
begin
  LSourceLength := ANumInputs;
  LTarget := 0;
  for i := Low(FNeurons) to High(FNeurons) do
  begin
    LSource := i;
    LTarget := (i+1);
    for k := Low(FNeurons[i]) to High(FNeurons[i]) do
      ABuffer[LTarget][k] := FNeurons[i, k].FeedForward(ABuffer[LSource]);
    LSourceLength := Length(FNeurons[i]);
  end;
  Result := Copy(ABuffer[LTarget], 0, LSourceLength);
end;

type
  TProtectedNeuron = class(TNeuron);

procedure TNeuralNetwork.Train(const AData: TArray<TArray<Single>>;
  const AExpected: TArray<Single>);
var
  LLearnRate: Single;
  LEpochs: Integer;
  i, k, m, o, p: Integer;
  LBuffer: TArray<TArray<Single>>;
  LResults: TArray<Single>;
  LDerivedBias, LPartialDerived, LDerived, LNextDerived: Single;
  LNeuron, LNextNeuron: TProtectedNeuron;
begin
  LLearnRate := 0.1;
  LEpochs := 1000;
  SetLength(LBuffer, Length(FNeurons) + 1);
  SetLength(LBuffer[0], FInputCount);
  for i := Low(LBuffer) + 1 to High(LBuffer) do
  begin
    SetLength(LBuffer[i], Length(FNeurons[i-1]));
  end;

  for i := 1 to LEpochs do
  begin
    for k := Low(AData) to High(AData) do
    begin
      for m := Low(AData[k]) to High(AData[k]) do
        LBuffer[0, m] := AData[k, m];
      LResults := InternalFeedForward(Length(AData[k]), LBuffer);

      LPartialDerived := -2 * (AExpected[k] - LResults[0]);

      for m := High(LBuffer) downto Low(LBuffer) + 1 do
      begin
        for o := Low(LBuffer[m]) to High(LBuffer[m]) do
        begin
          LNeuron := TProtectedNeuron(FNeurons[m-1, o]);
          LDerivedBias := Derive(LBuffer[m, o]);
          if o < High(LBuffer[m]) then
          begin
            LNextDerived := 0;
            for p := Low(FNeurons[m]) to High(FNeurons[m]) do
              LNextDerived := LNextDerived + TProtectedNeuron(FNeurons[m, p]).FWeights[o] * Derive(LBuffer[m+1, p]);
          end
          else
            LNextDerived := 1;
          LNeuron.FBias := LNeuron.FBias - LLearnRate * LPartialDerived * LDerivedBias * LNextDerived;
          for p := Low(LBuffer[m-1]) to High(LBuffer[m-1]) do
          begin
            LDerived := LBuffer[m-1, p] * LDerivedBias;
            LNeuron.FWeights[p] := LNeuron.FWeights[p] - LLearnRate * LPartialDerived * LDerived * LNextDerived;
          end;


        end;

      end;
    end;
    if ((i mod 10) = 0) and Assigned(FOnEpochProgression) then
      FOnEpochProgression(i);
  end;
end;

{ TNeuralNetwork<T> }

function TNeuralNetwork<T>.GetNeuronClass: TNeuronClass;
begin
  Result := T;
end;

end.
