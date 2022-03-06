unit Neural.Layers.Dense;

interface

uses
  Neural,
  Neural.Layers;

type
  TDenseLayer = class(TNeuralLayer)
  private
    FNeuronCount: Integer;
    FWeights: TArray<TArray<Single>>;
    FBiases: TArray<Single>;
    FLastOutput: TArray<Single>;
    FLastInput: TArray<Single>;
    function RunNode(const AInputs, AWeights: TArray<Single>; ABias: Single): Single;
    procedure RunNodeBackwards(const AResult, Gradient, ALearnRate: Single; var AWeights: TArray<Single>; var ABias: Single; var TargetGradients: TArray<Single>);
  public
    constructor Create(ACount: Integer; const AActivation: TActivation); reintroduce;
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; override;

  end;

implementation

uses
  Windows;

{ TDenseLayer }

function TDenseLayer.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<Single>;
var
  i: Integer;
begin
  SetLength(Result, Length(FLastInput));
  ZeroMemory(@Result[0], Length(Result) * SizeOf(Result[0]));
  for i := Low(FWeights) to High(FWeights) do
  begin
    RunNodeBackwards(FLastOutput[i], AGradients[i], ALearningRate, FWeights[i], FBiases[i], Result);
  end;
end;

procedure TDenseLayer.Build;
var
  i: Integer;
  LInputs: Integer;
begin
  inherited;
  SetLength(FWeights, FNeuronCount);
  SetLength(FBiases, FNeuronCount);
  FOutputShape := TShape.Create(FNeuronCount);
  LInputs := FInputShape.Size;
  for i := Low(FWeights) to High(FWeights) do
  begin
    FWeights[i] := BuildWeights(LInputs);
    FBiases[i] := 0;
  end;
end;

constructor TDenseLayer.Create(ACount: Integer; const AActivation: TActivation);
begin
  inherited Create(AActivation);
  FNeuronCount := ACount;
end;

function TDenseLayer.FeedForward(const Input: TArray<Single>): TArray<Single>;
var
  i: Integer;
begin
  FLastInput := Input;
  SetLength(Result, FOutputShape.Size);
  SetLength(FLastOutput, FOutputShape.Size);
  for i := Low(FWeights) to High(FWeights) do
  begin
    FLastOutput[i] := RunNode(Input, FWeights[i], FBiases[i]);
    Result[i] := FActivation.Run(FLastOutput[i]);
  end;
end;

function TDenseLayer.RunNode(const AInputs, AWeights: TArray<Single>;
  ABias: Single): Single;
var
  i: Integer;
begin
  Result := ABias;
  for i := Low(AWeights) to High(AWeights) do
    Result := Result + AInputs[i] * AWeights[i];
end;

procedure TDenseLayer.RunNodeBackwards(
  const AResult, Gradient, ALearnRate: Single;
  var AWeights: TArray<Single>;
  var ABias: Single;
  var TargetGradients: TArray<Single>);
var
  i: Integer;
  LDerived, LGradient: Single;
begin
  LDerived := FActivation.Derive(AResult);
  for i := Low(AWeights) to High(AWeights) do
  begin
    LGradient := FLastInput[i] * LDerived;
    TargetGradients[i] := TargetGradients[i] + AWeights[i] * LDerived * Gradient;
    AWeights[i] := AWeights[i] - (ALearnRate * Gradient * LGradient);
  end;
  ABias := ABias - (ALearnRate * Gradient * LDerived);
end;

end.
