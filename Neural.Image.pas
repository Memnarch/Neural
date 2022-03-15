unit Neural.Image;

interface

uses
  Neural,
  System.Types,
  Graphics;

type
  TNeuralImage = class
  private
    FImage: TBitmap;
  public
    constructor Create();
    destructor Destroy; override;
    procedure LoadFromFile(const AFile: string);
    function ToLumenValues: TNums;
    procedure FromLumenValues(const ANums: TNums);
    function ToRGB: TNums;
    procedure FromRGB(const Values: TNums);
    property Image: TBitmap read FImage;
  end;

implementation

uses
  System.Math,
  VCL.Imaging.jpeg;


type
  TRGB24 = packed record
    R, G, B: Byte;
  end;

  PRGB24 = ^TRGB24;

{$POINTERMATH ON}

{ TNeuralImage }

constructor TNeuralImage.Create;
begin
  inherited;
  FImage := TBitmap.Create(64, 64);
  FImage.PixelFormat := pf24bit;
end;

destructor TNeuralImage.Destroy;
begin
  FImage.Free;
  inherited;
end;

procedure TNeuralImage.FromLumenValues(const ANums: TNums);
var
  LLine: PRGB24;
  i, k: Integer;
  LVal: Byte;
  LNum: Single;
begin
  FImage.SetSize(ANums.Shape.Width, ANums.Shape.Height);
  for i := 0 to Pred(FImage.Height) do
  begin
    LLine := FImage.ScanLine[i];
    for k := 0 to Pred(FImage.Width) do
    begin
      LNum := ANums[k, i] * 255;
      LVal := Max(0, Min(255, Round(LNum)));
      LLine[k].R := LVal;
      LLine[k].G := LVal;
      LLine[k].B := LVal;
    end;
  end;
end;

procedure TNeuralImage.FromRGB(const Values: TNums);
var
  LLine: PRGB24;
  i, k: Integer;
begin
  FImage.SetSize(Values.Shape.Width, Values.Shape.Height);
  for i := 0 to Pred(FImage.Height) do
  begin
    LLine := FImage.ScanLine[i];
    for k := 0 to Pred(FImage.Width) do
    begin
      LLine[k].R := Max(0, Min(255, Round(Values[i, k, 0] * 255)));
      LLine[k].G := Max(0, Min(255, Round(Values[i, k, 1] * 255)));
      LLine[k].B := Max(0, Min(255, Round(Values[i, k, 2] * 255)));
    end;
  end;
end;

procedure TNeuralImage.LoadFromFile(const AFile: string);
var
  LPic: TPicture;
  LSize, LLeft, LTop: Integer;
  LScale: Single;
  LRect: TRect;
begin
  LPic := TPicture.Create();
  try
    LPic.LoadFromFile(AFile);
    FImage.Canvas.Brush.Color := clBlack;
    FImage.Canvas.FillRect(FImage.Canvas.ClipRect);
    if LPic.Width > LPic.Height then
      LSize := LPic.Width
    else
      LSize := LPic.Height;

    LScale := FImage.Width / LSize;
    LRect := TRect.Create(Point(0, 0), Round(LPic.Width*LScale), Round(LPic.Height*LScale));
    LLeft := (FImage.Width - LRect.Width) div 2;
    LTop := (FImage.Height - LRect.Height) div 2;
    LRect.Offset(LLeft, LTop);
    FImage.Canvas.StretchDraw(LRect, LPic.Graphic);
  finally
    LPic.Free;
  end;
end;

function TNeuralImage.ToLumenValues: TNums;
var
  LPixels: PRGB24;
  i, k: Integer;
  LLumen: Single;
begin
  Result := TNums.Create([Fimage.Width, FImage.Height]);
  for i := 0 to Pred(FImage.Height) do
  begin
    LPixels := FImage.ScanLine[i];
    for k := 0 to Pred(FImage.Width) do
    begin
      LLumen := 0.2126 * LPixels[k].R + 0.7152 * LPixels[k].G + 0.0722 * LPixels[k].B;
      Result[k, i] := LLumen / 255;//normalize
    end;
  end;
end;

function TNeuralImage.ToRGB: TNums;
var
  LPixels: PRGB24;
  i, k: Integer;
begin
  Result := TNums.Create([Fimage.Width, FImage.Height, 3]);
  for i := 0 to Pred(FImage.Height) do
  begin
    LPixels := FImage.ScanLine[i];
    for k := 0 to Pred(FImage.Width) do
    begin
      Result[k, i, 0] := LPixels[k].R / 255;//normalize
      Result[k, i, 1] := LPixels[k].G / 255;//normalize
      Result[k, i, 2] := LPixels[k].B / 255;//normalize
    end;
  end;
end;

end.
