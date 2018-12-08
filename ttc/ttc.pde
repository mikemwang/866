void setup(){
  size(500,500,P2D);
  noSmooth();
  noStroke();
}



void draw(){
  float t = frameCount * 0.3;
  background(0);
  fill(255);
  ellipse(width/2, height/2, 50 + t, 50 + t);
  saveFrame("frames/####.tif");
}
