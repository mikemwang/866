void setup() {
  size(500, 500, P3D);
  noSmooth();
  noStroke();
  frameRate(30);
}

ArrayList<String> ground_truth = new ArrayList<String>();

PVector goal = new PVector(100,100,0);
PVector start = new PVector(-100,-100, -1000);
PVector dir = PVector.sub(goal,start);
float goal_frames = 400;
PVector cur = start;

void draw() {
  float t = frameCount/goal_frames;
  cur = PVector.add(PVector.mult(dir, t), start);
  
  
  background(0);
  rectMode(CENTER);
  pushMatrix();
  translate(cur.x, cur.y, cur.z);
  
  for (int i = 0; i < 20; i++){
    fill(0);
    if (i%2 == 0){
      fill(255);
    }
    //fill(255-10*i);
    ellipse(width/2, height/2, width-25*i, height-25*i);
    
    //rect(width/2, 0, width-25*i, height-25*i);
  }
  //for (int j = 0; j < 10; j++){
  //  pushMatrix();
  //  translate(width/2, height/2);
  //  rotate(j*PI/10.0);
  //  rectMode(CENTER);
  //  fill(0);
  //  rect(0,0,width, 25);
  //  popMatrix();
  //}
  

  //rect(width/2, height/2, width, height);
  if (cur.z > 0) {
    exit();
  }
  popMatrix();
  saveFrame("frames/####.tif");
}
