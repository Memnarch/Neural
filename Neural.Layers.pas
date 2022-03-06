unit Neural.Layers;

interface

uses
  Neural;

type
  TNeuralLayer = class
  private
    FWeights: TArray<Single>;
  protected
    FInputShape: TShape;
    FOutputShape: TShape;
    FActivation: TActivation;
    function BuildWeights(const ACount: Integer): TArray<Single>; virtual;
  public
    constructor Create(const AActivation: TActivation); reintroduce;
    procedure Build; virtual; abstract;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; virtual; abstract;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; virtual; abstract;
    property InputShape: TShape read FInputShape write FInputShape;
    property OutputShape:TShape read FOutputShape;
    property Weights: TArray<Single> read FWeights;
  end;


implementation

function BuildWeights(const ACount: Integer): TArray<Single>;
var
  i: Integer;
  LWeight: Single;
begin
  Result := nil;
  SetLength(Result, ACount);
  LWeight := 1 / ACount;
  for i := 0 to Pred(ACount) do
    Result[i] := LWeight;
end;



{ TNeuralLayer }

function TNeuralLayer.BuildWeights(const ACount: Integer): TArray<Single>;
var
  LWeight: Single;
  i: Integer;
begin
  Result := nil;
  SetLength(Result, ACount);
  LWeight := 1 / ACount;
  for i := Low(Result) to High(Result) do
    Result[i] := LWeight;
end;

constructor TNeuralLayer.Create(const AActivation: TActivation);
begin
  inherited Create();
  FActivation := AActivation;
end;

end.
