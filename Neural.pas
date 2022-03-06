unit Neural;

interface

type
  TShape = packed record
  private
    function GetSize: Integer;
  public
    Width: Integer;
    Height: Integer;
    Depth: Integer;
    constructor Create(AWidth: Integer; AHeight: Integer = 1; ADepth: Integer = 1);
    property Size: Integer read GetSize;
  end;

  TActivationFunc = function(const Value: Single): Single;

  TActivation = packed record
    Run: TActivationFunc;
    Derive: TActivationFunc;
    class function Sigmoid: TActivation; static;
    class function Relu: TActivation; static;
  end;

implementation

uses
  Neural.Math;

{ TShape }

{ TShape }

constructor TShape.Create(AWidth, AHeight, ADepth: Integer);
begin
  Width := AWidth;
  Height := AHeight;
  Depth := ADepth;
end;

function TShape.GetSize: Integer;
begin
  Result := Width * Height * Depth;
end;


{ TActivation }

class function TActivation.Relu: TActivation;
begin
  Result.Run := Neural.Math.ReLu;
  Result.Derive := Neural.Math.DerivedReLu;
end;

class function TActivation.Sigmoid: TActivation;
begin
  Result.Run := Neural.Math.Sigmoid;
  Result.Derive := Neural.Math.DerivedSigmoid;
end;

end.
