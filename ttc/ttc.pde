void setup(){
  size(500,500,P3D);
  noSmooth();
  noStroke();
}



void draw(){
  float t = 3*frameCount;
  background(0);
  fill(255);
  rectMode(CENTER);
  pushMatrix();
  translate(0, 0, -1000+t);
  ellipse(width/2, height/2, 200, 200);
  popMatrix();
  saveFrame("frames/####.tif");
}
