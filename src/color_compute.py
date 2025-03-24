normal = {
    "input": r'''--primary-background: 255 255 255;
--primary-text: 17 24 39;
--primary-accent: 109 40 217;
--secondary-background: 243 244 246;
--secondary-text: 107 114 128;
--secondary-accent: 239 68 68;''',
    "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 255 255 255;
  --primary-background-hover: 242 242 242;
  --primary-background-focus: 230 230 230;
  --primary-background-disabled: 255 255 255;
  --primary-background-border: 209 209 209;
  --primary-background-gradient-a: 255 255 255;
  --primary-background-gradient-b: 245 245 245;
  --primary-background-gradient-c: 235 235 235;
  --primary-background-gradient-d: 220 220 220;

  --primary-text: 17 24 39;
  --primary-text-hover: 27 34 49;
  --primary-text-focus: 37 44 59;
  --primary-text-disabled: 87 94 109;
  --primary-text-border: 10 17 32;
  --primary-text-gradient-a: 17 24 39;
  --primary-text-gradient-b: 27 34 49;
  --primary-text-gradient-c: 37 44 59;
  --primary-text-gradient-d: 47 54 69;

  --primary-accent: 109 40 217;
  --primary-accent-hover: 98 36 195;
  --primary-accent-focus: 87 32 173;
  --primary-accent-disabled: 147 92 224;
  --primary-accent-border: 88 32 174;
  --primary-accent-gradient-a: 109 40 217;
  --primary-accent-gradient-b: 98 36 195;
  --primary-accent-gradient-c: 87 32 173;
  --primary-accent-gradient-d: 76 28 152;

  /* Secondary */
  --secondary-background: 243 244 246;
  --secondary-background-hover: 231 232 234;
  --secondary-background-focus: 219 220 221;
  --secondary-background-disabled: 243 244 246;
  --secondary-background-border: 199 200 202;
  --secondary-background-gradient-a: 243 244 246;
  --secondary-background-gradient-b: 231 232 234;
  --secondary-background-gradient-c: 219 220 221;
  --secondary-background-gradient-d: 207 208 209;

  --secondary-text: 107 114 128;
  --secondary-text-hover: 117 124 138;
  --secondary-text-focus: 127 134 148;
  --secondary-text-disabled: 150 157 171;
  --secondary-text-border: 91 97 109;
  --secondary-text-gradient-a: 107 114 128;
  --secondary-text-gradient-b: 117 124 138;
  --secondary-text-gradient-c: 127 134 148;
  --secondary-text-gradient-d: 137 144 158;

  --secondary-accent: 239 68 68;
  --secondary-accent-hover: 215 61 61;
  --secondary-accent-focus: 191 54 54;
  --secondary-accent-disabled: 242 120 120;
  --secondary-accent-border: 191 54 54;
  --secondary-accent-gradient-a: 239 68 68;
  --secondary-accent-gradient-b: 215 61 61;
  --secondary-accent-gradient-c: 191 54 54;
  --secondary-accent-gradient-d: 167 47 47;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 24 27 33;
  --primary-background-hover: 34 37 43;
  --primary-background-focus: 44 47 53;
  --primary-background-disabled: 19 22 28;
  --primary-background-border: 16 19 25;
  --primary-background-gradient-a: 24 27 33;
  --primary-background-gradient-b: 30 33 39;
  --primary-background-gradient-c: 36 39 45;
  --primary-background-gradient-d: 42 45 51;

  --primary-text: 230 232 235;
  --primary-text-hover: 240 242 245;
  --primary-text-focus: 250 252 255;
  --primary-text-disabled: 180 182 185;
  --primary-text-border: 210 212 215;
  --primary-text-gradient-a: 230 232 235;
  --primary-text-gradient-b: 238 240 243;
  --primary-text-gradient-c: 246 248 251;
  --primary-text-gradient-d: 255 255 255;

  --primary-accent: 138 66 255;
  --primary-accent-hover: 156 92 255;
  --primary-accent-focus: 174 118 255;
  --primary-accent-disabled: 101 57 183;
  --primary-accent-border: 120 58 220;
  --primary-accent-gradient-a: 138 66 255;
  --primary-accent-gradient-b: 150 80 255;
  --primary-accent-gradient-c: 162 94 255;
  --primary-accent-gradient-d: 174 108 255;

  /* Secondary */
  --secondary-background: 36 39 45;
  --secondary-background-hover: 46 49 55;
  --secondary-background-focus: 56 59 65;
  --secondary-background-disabled: 31 34 40;
  --secondary-background-border: 28 31 37;
  --secondary-background-gradient-a: 36 39 45;
  --secondary-background-gradient-b: 42 45 51;
  --secondary-background-gradient-c: 48 51 57;
  --secondary-background-gradient-d: 54 57 63;

  --secondary-text: 190 196 204;
  --secondary-text-hover: 200 206 214;
  --secondary-text-focus: 210 216 224;
  --secondary-text-disabled: 150 156 164;
  --secondary-text-border: 170 176 184;
  --secondary-text-gradient-a: 190 196 204;
  --secondary-text-gradient-b: 200 206 214;
  --secondary-text-gradient-c: 210 216 224;
  --secondary-text-gradient-d: 220 226 234;

  --secondary-accent: 255 87 87;
  --secondary-accent-hover: 255 107 107;
  --secondary-accent-focus: 255 127 127;
  --secondary-accent-disabled: 184 63 63;
  --secondary-accent-border: 223 76 76;
  --secondary-accent-gradient-a: 255 87 87;
  --secondary-accent-gradient-b: 255 102 102;
  --secondary-accent-gradient-c: 255 117 117;
  --secondary-accent-gradient-d: 255 132 132;
}'''
}

monochromatic = {
  "input": r'''--primary-background: 240 253 250;
--primary-text: 0 95 90;
--primary-accent: 0 150 137;
--secondary-background: 150 247 228;
--secondary-text: 0 120 111;
--secondary-accent: 0 150 137;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 240 253 250;
  --primary-background-hover: 228 240 238;
  --primary-background-focus: 216 228 225;
  --primary-background-disabled: 240 253 250;
  --primary-background-border: 197 207 205;
  --primary-background-gradient-a: 240 253 250;
  --primary-background-gradient-b: 232 245 242;
  --primary-background-gradient-c: 224 236 233;
  --primary-background-gradient-d: 204 215 213;

  --primary-text: 0 95 90;
  --primary-text-hover: 0 90 86;
  --primary-text-focus: 0 85 81;
  --primary-text-disabled: 60 129 122;
  --primary-text-border: 0 81 77;
  --primary-text-gradient-a: 0 95 90;
  --primary-text-gradient-b: 0 90 86;
  --primary-text-gradient-c: 0 85 81;
  --primary-text-gradient-d: 0 80 76;

  --primary-accent: 0 150 137;
  --primary-accent-hover: 0 143 130;
  --primary-accent-focus: 0 135 123;
  --primary-accent-disabled: 60 170 157;
  --primary-accent-border: 0 128 116;
  --primary-accent-gradient-a: 0 150 137;
  --primary-accent-gradient-b: 0 143 130;
  --primary-accent-gradient-c: 0 135 123;
  --primary-accent-gradient-d: 0 128 116;

  /* Secondary */
  --secondary-background: 150 247 228;
  --secondary-background-hover: 143 235 217;
  --secondary-background-focus: 135 222 205;
  --secondary-background-disabled: 150 247 228;
  --secondary-background-border: 123 203 187;
  --secondary-background-gradient-a: 150 247 228;
  --secondary-background-gradient-b: 145 239 221;
  --secondary-background-gradient-c: 138 228 211;
  --secondary-background-gradient-d: 128 210 194;

  --secondary-text: 0 120 111;
  --secondary-text-hover: 0 114 105;
  --secondary-text-focus: 0 108 100;
  --secondary-text-disabled: 38 145 135;
  --secondary-text-border: 0 102 94;
  --secondary-text-gradient-a: 0 120 111;
  --secondary-text-gradient-b: 0 114 105;
  --secondary-text-gradient-c: 0 108 100;
  --secondary-text-gradient-d: 0 102 94;

  --secondary-accent: 0 150 137;
  --secondary-accent-hover: 0 143 130;
  --secondary-accent-focus: 0 135 123;
  --secondary-accent-disabled: 38 172 159;
  --secondary-accent-border: 0 128 116;
  --secondary-accent-gradient-a: 0 150 137;
  --secondary-accent-gradient-b: 0 143 130;
  --secondary-accent-gradient-c: 0 135 123;
  --secondary-accent-gradient-d: 0 128 116;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 20 40 38;
  --primary-background-hover: 30 50 48;
  --primary-background-focus: 40 60 58;
  --primary-background-disabled: 15 35 33;
  --primary-background-border: 12 30 28;
  --primary-background-gradient-a: 20 40 38;
  --primary-background-gradient-b: 25 45 43;
  --primary-background-gradient-c: 30 50 48;
  --primary-background-gradient-d: 35 55 53;

  --primary-text: 180 240 235;
  --primary-text-hover: 190 250 245;
  --primary-text-focus: 200 255 250;
  --primary-text-disabled: 140 200 195;
  --primary-text-border: 190 250 245;
  --primary-text-gradient-a: 180 240 235;
  --primary-text-gradient-b: 190 250 245;
  --primary-text-gradient-c: 200 255 250;
  --primary-text-gradient-d: 210 255 255;

  --primary-accent: 20 180 167;
  --primary-accent-hover: 40 195 182;
  --primary-accent-focus: 60 210 197;
  --primary-accent-disabled: 15 135 125;
  --primary-accent-border: 18 160 147;
  --primary-accent-gradient-a: 20 180 167;
  --primary-accent-gradient-b: 40 195 182;
  --primary-accent-gradient-c: 60 210 197;
  --primary-accent-gradient-d: 80 225 212;

  /* Secondary */
  --secondary-background: 30 50 48;
  --secondary-background-hover: 40 60 58;
  --secondary-background-focus: 50 70 68;
  --secondary-background-disabled: 25 45 43;
  --secondary-background-border: 22 40 38;
  --secondary-background-gradient-a: 30 50 48;
  --secondary-background-gradient-b: 35 55 53;
  --secondary-background-gradient-c: 40 60 58;
  --secondary-background-gradient-d: 45 65 63;

  --secondary-text: 160 220 215;
  --secondary-text-hover: 170 230 225;
  --secondary-text-focus: 180 240 235;
  --secondary-text-disabled: 120 180 175;
  --secondary-text-border: 170 230 225;
  --secondary-text-gradient-a: 160 220 215;
  --secondary-text-gradient-b: 170 230 225;
  --secondary-text-gradient-c: 180 240 235;
  --secondary-text-gradient-d: 190 250 245;

  --secondary-accent: 20 180 167;
  --secondary-accent-hover: 40 195 182;
  --secondary-accent-focus: 60 210 197;
  --secondary-accent-disabled: 15 135 125;
  --secondary-accent-border: 18 160 147;
  --secondary-accent-gradient-a: 20 180 167;
  --secondary-accent-gradient-b: 40 195 182;
  --secondary-accent-gradient-c: 60 210 197;
  --secondary-accent-gradient-d: 80 225 212;
}'''
}

neutral_with_accent = {
  "input": r'''--primary-background: 248 250 252;
--primary-text: 15 23 43;
--primary-accent: 0 201 80;
--secondary-background: 241 245 249;
--secondary-text: 29 41 61;
--secondary-accent: 0 166 62;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 248 250 252;
  --primary-background-hover: 238 240 242;
  --primary-background-focus: 223 225 227;
  --primary-background-disabled: 248 250 252;
  --primary-background-border: 203 205 207;
  --primary-background-gradient-a: 248 250 252;
  --primary-background-gradient-b: 238 240 242;
  --primary-background-gradient-c: 228 230 232;
  --primary-background-gradient-d: 208 210 212;

  --primary-text: 15 23 43;
  --primary-text-hover: 25 33 53;
  --primary-text-focus: 35 43 63;
  --primary-text-disabled: 75 83 103;
  --primary-text-border: 8 16 36;
  --primary-text-gradient-a: 15 23 43;
  --primary-text-gradient-b: 25 33 53;
  --primary-text-gradient-c: 35 43 63;
  --primary-text-gradient-d: 45 53 73;

  --primary-accent: 0 201 80;
  --primary-accent-hover: 0 181 72;
  --primary-accent-focus: 0 161 64;
  --primary-accent-disabled: 80 221 140;
  --primary-accent-border: 0 161 64;
  --primary-accent-gradient-a: 0 201 80;
  --primary-accent-gradient-b: 0 181 72;
  --primary-accent-gradient-c: 0 161 64;
  --primary-accent-gradient-d: 0 141 56;

  /* Secondary */
  --secondary-background: 241 245 249;
  --secondary-background-hover: 229 233 237;
  --secondary-background-focus: 217 221 225;
  --secondary-background-disabled: 241 245 249;
  --secondary-background-border: 197 201 205;
  --secondary-background-gradient-a: 241 245 249;
  --secondary-background-gradient-b: 231 235 239;
  --secondary-background-gradient-c: 221 225 229;
  --secondary-background-gradient-d: 201 205 209;

  --secondary-text: 29 41 61;
  --secondary-text-hover: 39 51 71;
  --secondary-text-focus: 49 61 81;
  --secondary-text-disabled: 90 102 122;
  --secondary-text-border: 23 35 55;
  --secondary-text-gradient-a: 29 41 61;
  --secondary-text-gradient-b: 39 51 71;
  --secondary-text-gradient-c: 49 61 81;
  --secondary-text-gradient-d: 59 71 91;

  --secondary-accent: 0 166 62;
  --secondary-accent-hover: 0 149 56;
  --secondary-accent-focus: 0 133 50;
  --secondary-accent-disabled: 70 186 112;
  --secondary-accent-border: 0 133 50;
  --secondary-accent-gradient-a: 0 166 62;
  --secondary-accent-gradient-b: 0 149 56;
  --secondary-accent-gradient-c: 0 133 50;
  --secondary-accent-gradient-d: 0 116 44;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 22 25 31;
  --primary-background-hover: 32 35 41;
  --primary-background-focus: 42 45 51;
  --primary-background-disabled: 17 20 26;
  --primary-background-border: 15 18 24;
  --primary-background-gradient-a: 22 25 31;
  --primary-background-gradient-b: 28 31 37;
  --primary-background-gradient-c: 34 37 43;
  --primary-background-gradient-d: 40 43 49;

  --primary-text: 235 237 245;
  --primary-text-hover: 245 247 255;
  --primary-text-focus: 255 255 255;
  --primary-text-disabled: 185 187 195;
  --primary-text-border: 215 217 225;
  --primary-text-gradient-a: 235 237 245;
  --primary-text-gradient-b: 243 245 253;
  --primary-text-gradient-c: 251 253 255;
  --primary-text-gradient-d: 255 255 255;

  --primary-accent: 20 231 100;
  --primary-accent-hover: 40 251 120;
  --primary-accent-focus: 60 255 140;
  --primary-accent-disabled: 18 185 80;
  --primary-accent-border: 16 208 90;
  --primary-accent-gradient-a: 20 231 100;
  --primary-accent-gradient-b: 35 241 110;
  --primary-accent-gradient-c: 50 251 120;
  --primary-accent-gradient-d: 65 255 130;

  /* Secondary */
  --secondary-background: 32 35 41;
  --secondary-background-hover: 42 45 51;
  --secondary-background-focus: 52 55 61;
  --secondary-background-disabled: 27 30 36;
  --secondary-background-border: 25 28 34;
  --secondary-background-gradient-a: 32 35 41;
  --secondary-background-gradient-b: 38 41 47;
  --secondary-background-gradient-c: 44 47 53;
  --secondary-background-gradient-d: 50 53 59;

  --secondary-text: 210 215 225;
  --secondary-text-hover: 220 225 235;
  --secondary-text-focus: 230 235 245;
  --secondary-text-disabled: 170 175 185;
  --secondary-text-border: 190 195 205;
  --secondary-text-gradient-a: 210 215 225;
  --secondary-text-gradient-b: 220 225 235;
  --secondary-text-gradient-c: 230 235 245;
  --secondary-text-gradient-d: 240 245 255;

  --secondary-accent: 20 196 82;
  --secondary-accent-hover: 40 216 102;
  --secondary-accent-focus: 60 236 122;
  --secondary-accent-disabled: 16 157 66;
  --secondary-accent-border: 18 177 74;
  --secondary-accent-gradient-a: 20 196 82;
  --secondary-accent-gradient-b: 35 211 92;
  --secondary-accent-gradient-c: 50 226 102;
  --secondary-accent-gradient-d: 65 241 112;
}'''
}

neutral_with_accent2 = {
    "input": r'''--primary-background: 250 250 249;
--primary-text: 28 25 23;
--primary-accent: 240 177 0;
--secondary-background: 245 245 244;
--secondary-text: 41 37 36;
--secondary-accent: 208 135 0;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 250 250 249;
  --primary-background-hover: 240 240 239;
  --primary-background-focus: 225 225 224;
  --primary-background-disabled: 250 250 249;
  --primary-background-border: 205 205 204;
  --primary-background-gradient-a: 250 250 249;
  --primary-background-gradient-b: 240 240 239;
  --primary-background-gradient-c: 225 225 224;
  --primary-background-gradient-d: 205 205 204;

  --primary-text: 28 25 23;
  --primary-text-hover: 38 35 33;
  --primary-text-focus: 48 45 43;
  --primary-text-disabled: 88 85 83;
  --primary-text-border: 18 15 13;
  --primary-text-gradient-a: 28 25 23;
  --primary-text-gradient-b: 38 35 33;
  --primary-text-gradient-c: 48 45 43;
  --primary-text-gradient-d: 58 55 53;

  --primary-accent: 240 177 0;
  --primary-accent-hover: 216 159 0;
  --primary-accent-focus: 192 141 0;
  --primary-accent-disabled: 245 200 75;
  --primary-accent-border: 192 141 0;
  --primary-accent-gradient-a: 240 177 0;
  --primary-accent-gradient-b: 216 159 0;
  --primary-accent-gradient-c: 192 141 0;
  --primary-accent-gradient-d: 168 123 0;

  /* Secondary */
  --secondary-background: 245 245 244;
  --secondary-background-hover: 230 230 229;
  --secondary-background-focus: 215 215 214;
  --secondary-background-disabled: 245 245 244;
  --secondary-background-border: 200 200 199;
  --secondary-background-gradient-a: 245 245 244;
  --secondary-background-gradient-b: 230 230 229;
  --secondary-background-gradient-c: 215 215 214;
  --secondary-background-gradient-d: 200 200 199;

  --secondary-text: 41 37 36;
  --secondary-text-hover: 51 47 46;
  --secondary-text-focus: 61 57 56;
  --secondary-text-disabled: 101 97 96;
  --secondary-text-border: 31 27 26;
  --secondary-text-gradient-a: 41 37 36;
  --secondary-text-gradient-b: 51 47 46;
  --secondary-text-gradient-c: 61 57 56;
  --secondary-text-gradient-d: 71 67 66;

  --secondary-accent: 208 135 0;
  --secondary-accent-hover: 187 122 0;
  --secondary-accent-focus: 166 108 0;
  --secondary-accent-disabled: 220 170 75;
  --secondary-accent-border: 166 108 0;
  --secondary-accent-gradient-a: 208 135 0;
  --secondary-accent-gradient-b: 187 122 0;
  --secondary-accent-gradient-c: 166 108 0;
  --secondary-accent-gradient-d: 145 94 0;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 25 25 26;
  --primary-background-hover: 35 35 36;
  --primary-background-focus: 45 45 46;
  --primary-background-disabled: 20 20 21;
  --primary-background-border: 15 15 16;
  --primary-background-gradient-a: 25 25 26;
  --primary-background-gradient-b: 32 32 33;
  --primary-background-gradient-c: 39 39 40;
  --primary-background-gradient-d: 46 46 47;

  --primary-text: 230 233 235;
  --primary-text-hover: 240 243 245;
  --primary-text-focus: 250 253 255;
  --primary-text-disabled: 180 183 185;
  --primary-text-border: 210 213 215;
  --primary-text-gradient-a: 230 233 235;
  --primary-text-gradient-b: 238 241 243;
  --primary-text-gradient-c: 246 249 251;
  --primary-text-gradient-d: 255 255 255;

  --primary-accent: 255 207 30;
  --primary-accent-hover: 255 217 60;
  --primary-accent-focus: 255 227 90;
  --primary-accent-disabled: 204 156 24;
  --primary-accent-border: 230 182 27;
  --primary-accent-gradient-a: 255 207 30;
  --primary-accent-gradient-b: 255 217 60;
  --primary-accent-gradient-c: 255 227 90;
  --primary-accent-gradient-d: 255 237 120;

  /* Secondary */
  --secondary-background: 35 35 36;
  --secondary-background-hover: 45 45 46;
  --secondary-background-focus: 55 55 56;
  --secondary-background-disabled: 30 30 31;
  --secondary-background-border: 25 25 26;
  --secondary-background-gradient-a: 35 35 36;
  --secondary-background-gradient-b: 42 42 43;
  --secondary-background-gradient-c: 49 49 50;
  --secondary-background-gradient-d: 56 56 57;

  --secondary-text: 210 214 215;
  --secondary-text-hover: 220 224 225;
  --secondary-text-focus: 230 234 235;
  --secondary-text-disabled: 160 164 165;
  --secondary-text-border: 190 194 195;
  --secondary-text-gradient-a: 210 214 215;
  --secondary-text-gradient-b: 220 224 225;
  --secondary-text-gradient-c: 230 234 235;
  --secondary-text-gradient-d: 240 244 245;

  --secondary-accent: 235 162 27;
  --secondary-accent-hover: 245 182 47;
  --secondary-accent-focus: 255 202 67;
  --secondary-accent-disabled: 188 129 22;
  --secondary-accent-border: 210 137 24;
  --secondary-accent-gradient-a: 235 162 27;
  --secondary-accent-gradient-b: 245 182 47;
  --secondary-accent-gradient-c: 255 202 67;
  --secondary-accent-gradient-d: 255 222 87;
}
'''
}

analogous = {
    "input": r'''--primary-background: 254 242 242;
--primary-text: 130 24 26;
--primary-accent: 255 32 86;
--secondary-background: 255 226 226;
--secondary-text: 159 7 18;
--secondary-accent: 245 73 0;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 254 242 242;
  --primary-background-hover: 242 230 230;
  --primary-background-focus: 230 218 218;
  --primary-background-disabled: 254 242 242;
  --primary-background-border: 208 199 199;
  --primary-background-gradient-a: 254 242 242;
  --primary-background-gradient-b: 248 236 236;
  --primary-background-gradient-c: 230 218 218;
  --primary-background-gradient-d: 208 199 199;

  --primary-text: 130 24 26;
  --primary-text-hover: 140 34 36;
  --primary-text-focus: 150 44 46;
  --primary-text-disabled: 167 87 88;
  --primary-text-border: 110 4 6;
  --primary-text-gradient-a: 130 24 26;
  --primary-text-gradient-b: 140 34 36;
  --primary-text-gradient-c: 150 44 46;
  --primary-text-gradient-d: 160 54 56;

  --primary-accent: 255 32 86;
  --primary-accent-hover: 230 29 77;
  --primary-accent-focus: 205 26 68;
  --primary-accent-disabled: 255 84 126;
  --primary-accent-border: 204 26 69;
  --primary-accent-gradient-a: 255 32 86;
  --primary-accent-gradient-b: 230 29 77;
  --primary-accent-gradient-c: 205 26 68;
  --primary-accent-gradient-d: 180 23 59;

  /* Secondary */
  --secondary-background: 255 226 226;
  --secondary-background-hover: 242 215 215;
  --secondary-background-focus: 230 204 204;
  --secondary-background-disabled: 255 226 226;
  --secondary-background-border: 209 186 186;
  --secondary-background-gradient-a: 255 226 226;
  --secondary-background-gradient-b: 242 215 215;
  --secondary-background-gradient-c: 230 204 204;
  --secondary-background-gradient-d: 209 186 186;

  --secondary-text: 159 7 18;
  --secondary-text-hover: 169 17 28;
  --secondary-text-focus: 179 27 38;
  --secondary-text-disabled: 185 67 74;
  --secondary-text-border: 139 0 4;
  --secondary-text-gradient-a: 159 7 18;
  --secondary-text-gradient-b: 169 17 28;
  --secondary-text-gradient-c: 179 27 38;
  --secondary-text-gradient-d: 189 37 48;

  --secondary-accent: 245 73 0;
  --secondary-accent-hover: 220 66 0;
  --secondary-accent-focus: 195 59 0;
  --secondary-accent-disabled: 248 116 56;
  --secondary-accent-border: 196 58 0;
  --secondary-accent-gradient-a: 245 73 0;
  --secondary-accent-gradient-b: 220 66 0;
  --secondary-accent-gradient-c: 195 59 0;
  --secondary-accent-gradient-d: 170 52 0;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 32 24 24;
  --primary-background-hover: 42 34 34;
  --primary-background-focus: 52 44 44;
  --primary-background-disabled: 27 19 19;
  --primary-background-border: 22 14 14;
  --primary-background-gradient-a: 32 24 24;
  --primary-background-gradient-b: 38 30 30;
  --primary-background-gradient-c: 44 36 36;
  --primary-background-gradient-d: 50 42 42;

  --primary-text: 255 180 181;
  --primary-text-hover: 255 190 191;
  --primary-text-focus: 255 200 201;
  --primary-text-disabled: 205 140 141;
  --primary-text-border: 235 160 161;
  --primary-text-gradient-a: 255 180 181;
  --primary-text-gradient-b: 255 190 191;
  --primary-text-gradient-c: 255 200 201;
  --primary-text-gradient-d: 255 210 211;

  --primary-accent: 255 62 116;
  --primary-accent-hover: 255 82 136;
  --primary-accent-focus: 255 102 156;
  --primary-accent-disabled: 204 50 93;
  --primary-accent-border: 230 56 104;
  --primary-accent-gradient-a: 255 62 116;
  --primary-accent-gradient-b: 255 82 136;
  --primary-accent-gradient-c: 255 102 156;
  --primary-accent-gradient-d: 255 122 176;

  /* Secondary */
  --secondary-background: 42 28 28;
  --secondary-background-hover: 52 38 38;
  --secondary-background-focus: 62 48 48;
  --secondary-background-disabled: 37 23 23;
  --secondary-background-border: 32 18 18;
  --secondary-background-gradient-a: 42 28 28;
  --secondary-background-gradient-b: 48 34 34;
  --secondary-background-gradient-c: 54 40 40;
  --secondary-background-gradient-d: 60 46 46;

  --secondary-text: 255 160 164;
  --secondary-text-hover: 255 170 174;
  --secondary-text-focus: 255 180 184;
  --secondary-text-disabled: 205 128 132;
  --secondary-text-border: 235 144 148;
  --secondary-text-gradient-a: 255 160 164;
  --secondary-text-gradient-b: 255 170 174;
  --secondary-text-gradient-c: 255 180 184;
  --secondary-text-gradient-d: 255 190 194;

  --secondary-accent: 255 103 30;
  --secondary-accent-hover: 255 123 50;
  --secondary-accent-focus: 255 143 70;
  --secondary-accent-disabled: 204 82 24;
  --secondary-accent-border: 230 93 27;
  --secondary-accent-gradient-a: 255 103 30;
  --secondary-accent-gradient-b: 255 123 50;
  --secondary-accent-gradient-c: 255 143 70;
  --secondary-accent-gradient-d: 255 163 90;
}
'''
}

complementary = {
    "input": r'''--primary-background: 239 246 255;
--primary-text: 28 57 142;
--primary-accent: 255 105 0;
--secondary-background: 219 234 254;
--secondary-text: 25 60 184;
--secondary-accent: 245 73 0;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 239 246 255;
  --primary-background-hover: 227 234 242;
  --primary-background-focus: 215 221 230;
  --primary-background-disabled: 239 246 255;
  --primary-background-border: 196 202 209;
  --primary-background-gradient-a: 239 246 255;
  --primary-background-gradient-b: 231 238 247;
  --primary-background-gradient-c: 223 230 239;
  --primary-background-gradient-d: 203 209 217;

  --primary-text: 28 57 142;
  --primary-text-hover: 27 54 135;
  --primary-text-focus: 25 51 128;
  --primary-text-disabled: 72 96 163;
  --primary-text-border: 23 47 116;
  --primary-text-gradient-a: 28 57 142;
  --primary-text-gradient-b: 27 54 135;
  --primary-text-gradient-c: 25 51 128;
  --primary-text-gradient-d: 23 47 116;

  --primary-accent: 255 105 0;
  --primary-accent-hover: 242 100 0;
  --primary-accent-focus: 229 95 0;
  --primary-accent-disabled: 255 138 51;
  --primary-accent-border: 209 86 0;
  --primary-accent-gradient-a: 255 105 0;
  --primary-accent-gradient-b: 242 100 0;
  --primary-accent-gradient-c: 229 95 0;
  --primary-accent-gradient-d: 209 86 0;

  /* Secondary */
  --secondary-background: 219 234 254;
  --secondary-background-hover: 208 222 241;
  --secondary-background-focus: 197 211 229;
  --secondary-background-disabled: 219 234 254;
  --secondary-background-border: 180 192 208;
  --secondary-background-gradient-a: 219 234 254;
  --secondary-background-gradient-b: 211 226 245;
  --secondary-background-gradient-c: 203 218 236;
  --secondary-background-gradient-d: 185 198 214;

  --secondary-text: 25 60 184;
  --secondary-text-hover: 24 57 175;
  --secondary-text-focus: 23 54 166;
  --secondary-text-disabled: 71 101 199;
  --secondary-text-border: 21 49 151;
  --secondary-text-gradient-a: 25 60 184;
  --secondary-text-gradient-b: 24 57 175;
  --secondary-text-gradient-c: 23 54 166;
  --secondary-text-gradient-d: 21 49 151;

  --secondary-accent: 245 73 0;
  --secondary-accent-hover: 233 69 0;
  --secondary-accent-focus: 221 66 0;
  --secondary-accent-disabled: 246 110 49;
  --secondary-accent-border: 201 60 0;
  --secondary-accent-gradient-a: 245 73 0;
  --secondary-accent-gradient-b: 233 69 0;
  --secondary-accent-gradient-c: 221 66 0;
  --secondary-accent-gradient-d: 201 60 0;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 28 32 43;
  --primary-background-hover: 38 42 53;
  --primary-background-focus: 48 52 63;
  --primary-background-disabled: 23 27 38;
  --primary-background-border: 20 24 35;
  --primary-background-gradient-a: 28 32 43;
  --primary-background-gradient-b: 35 39 50;
  --primary-background-gradient-c: 42 46 57;
  --primary-background-gradient-d: 49 53 64;

  --primary-text: 220 230 255;
  --primary-text-hover: 230 240 255;
  --primary-text-focus: 240 250 255;
  --primary-text-disabled: 180 190 230;
  --primary-text-border: 209 219 243;
  --primary-text-gradient-a: 220 230 255;
  --primary-text-gradient-b: 228 238 255;
  --primary-text-gradient-c: 236 246 255;
  --primary-text-gradient-d: 244 254 255;

  --primary-accent: 255 135 40;
  --primary-accent-hover: 255 150 60;
  --primary-accent-focus: 255 165 80;
  --primary-accent-disabled: 204 108 32;
  --primary-accent-border: 230 122 36;
  --primary-accent-gradient-a: 255 135 40;
  --primary-accent-gradient-b: 255 145 50;
  --primary-accent-gradient-c: 255 155 60;
  --primary-accent-gradient-d: 255 165 70;

  /* Secondary */
  --secondary-background: 40 45 58;
  --secondary-background-hover: 50 55 68;
  --secondary-background-focus: 60 65 78;
  --secondary-background-disabled: 35 40 53;
  --secondary-background-border: 32 37 50;
  --secondary-background-gradient-a: 40 45 58;
  --secondary-background-gradient-b: 47 52 65;
  --secondary-background-gradient-c: 54 59 72;
  --secondary-background-gradient-d: 61 66 79;

  --secondary-text: 210 220 255;
  --secondary-text-hover: 220 230 255;
  --secondary-text-focus: 230 240 255;
  --secondary-text-disabled: 170 180 230;
  --secondary-text-border: 200 210 243;
  --secondary-text-gradient-a: 210 220 255;
  --secondary-text-gradient-b: 218 228 255;
  --secondary-text-gradient-c: 226 236 255;
  --secondary-text-gradient-d: 234 244 255;

  --secondary-accent: 255 120 30;
  --secondary-accent-hover: 255 135 50;
  --secondary-accent-focus: 255 150 70;
  --secondary-accent-disabled: 204 96 24;
  --secondary-accent-border: 230 108 27;
  --secondary-accent-gradient-a: 255 120 30;
  --secondary-accent-gradient-b: 255 130 40;
  --secondary-accent-gradient-c: 255 140 50;
  --secondary-accent-gradient-d: 255 150 60;
}
'''
}

triadic = {
    "input": r'''--primary-background: 254 252 232;
--primary-text: 115 62 10;
--primary-accent: 0 184 219;
--secondary-background: 254 249 194;
--secondary-text: 137 75 0;
--secondary-accent: 200 0 222;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 254 252 232;
  --primary-background-hover: 241 239 220;
  --primary-background-focus: 229 227 209;
  --primary-background-disabled: 254 252 232;
  --primary-background-border: 203 201 186;
  --primary-background-gradient-a: 254 252 232;
  --primary-background-gradient-b: 247 245 226;
  --primary-background-gradient-c: 235 233 214;
  --primary-background-gradient-d: 213 211 195;

  --primary-text: 115 62 10;
  --primary-text-hover: 109 59 9;
  --primary-text-focus: 103 56 9;
  --primary-text-disabled: 143 90 38;
  --primary-text-border: 94 51 8;
  --primary-text-gradient-a: 115 62 10;
  --primary-text-gradient-b: 109 59 9;
  --primary-text-gradient-c: 103 56 9;
  --primary-text-gradient-d: 97 53 8;

  --primary-accent: 0 184 219;
  --primary-accent-hover: 0 175 208;
  --primary-accent-focus: 0 166 197;
  --primary-accent-disabled: 51 193 222;
  --primary-accent-border: 0 147 175;
  --primary-accent-gradient-a: 0 184 219;
  --primary-accent-gradient-b: 0 175 208;
  --primary-accent-gradient-c: 0 166 197;
  --primary-accent-gradient-d: 0 154 183;

  /* Secondary */
  --secondary-background: 254 249 194;
  --secondary-background-hover: 241 237 184;
  --secondary-background-focus: 229 224 175;
  --secondary-background-disabled: 254 249 194;
  --secondary-background-border: 203 199 155;
  --secondary-background-gradient-a: 254 249 194;
  --secondary-background-gradient-b: 247 243 189;
  --secondary-background-gradient-c: 235 230 179;
  --secondary-background-gradient-d: 213 208 162;

  --secondary-text: 137 75 0;
  --secondary-text-hover: 130 71 0;
  --secondary-text-focus: 123 67 0;
  --secondary-text-disabled: 164 104 29;
  --secondary-text-border: 111 61 0;
  --secondary-text-gradient-a: 137 75 0;
  --secondary-text-gradient-b: 130 71 0;
  --secondary-text-gradient-c: 123 67 0;
  --secondary-text-gradient-d: 116 64 0;

  --secondary-accent: 200 0 222;
  --secondary-accent-hover: 190 0 211;
  --secondary-accent-focus: 180 0 200;
  --secondary-accent-disabled: 214 51 227;
  --secondary-accent-border: 160 0 178;
  --secondary-accent-gradient-a: 200 0 222;
  --secondary-accent-gradient-b: 190 0 211;
  --secondary-accent-gradient-c: 180 0 200;
  --secondary-accent-gradient-d: 166 0 184;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 28 32 40;
  --primary-background-hover: 38 42 50;
  --primary-background-focus: 48 52 60;
  --primary-background-disabled: 23 27 35;
  --primary-background-border: 20 24 32;
  --primary-background-gradient-a: 28 32 40;
  --primary-background-gradient-b: 34 38 46;
  --primary-background-gradient-c: 40 44 52;
  --primary-background-gradient-d: 46 50 58;

  --primary-text: 235 225 210;
  --primary-text-hover: 245 235 220;
  --primary-text-focus: 255 245 230;
  --primary-text-disabled: 185 175 160;
  --primary-text-border: 215 205 190;
  --primary-text-gradient-a: 235 225 210;
  --primary-text-gradient-b: 242 232 217;
  --primary-text-gradient-c: 249 239 224;
  --primary-text-gradient-d: 255 246 231;

  --primary-accent: 25 204 239;
  --primary-accent-hover: 45 214 249;
  --primary-accent-focus: 65 224 255;
  --primary-accent-disabled: 20 163 191;
  --primary-accent-border: 22 184 215;
  --primary-accent-gradient-a: 25 204 239;
  --primary-accent-gradient-b: 35 214 249;
  --primary-accent-gradient-c: 45 224 255;
  --primary-accent-gradient-d: 55 234 255;

  /* Secondary */
  --secondary-background: 40 44 52;
  --secondary-background-hover: 50 54 62;
  --secondary-background-focus: 60 64 72;
  --secondary-background-disabled: 35 39 47;
  --secondary-background-border: 32 36 44;
  --secondary-background-gradient-a: 40 44 52;
  --secondary-background-gradient-b: 46 50 58;
  --secondary-background-gradient-c: 52 56 64;
  --secondary-background-gradient-d: 58 62 70;

  --secondary-text: 230 220 190;
  --secondary-text-hover: 240 230 200;
  --secondary-text-focus: 250 240 210;
  --secondary-text-disabled: 180 170 140;
  --secondary-text-border: 210 200 170;
  --secondary-text-gradient-a: 230 220 190;
  --secondary-text-gradient-b: 237 227 197;
  --secondary-text-gradient-c: 244 234 204;
  --secondary-text-gradient-d: 251 241 211;

  --secondary-accent: 220 30 242;
  --secondary-accent-hover: 230 50 252;
  --secondary-accent-focus: 240 70 255;
  --secondary-accent-disabled: 176 24 194;
  --secondary-accent-border: 198 27 218;
  --secondary-accent-gradient-a: 220 30 242;
  --secondary-accent-gradient-b: 227 40 247;
  --secondary-accent-gradient-c: 234 50 252;
  --secondary-accent-gradient-d: 241 60 255;
}
'''
}

split_complementary = {
    "input": r'''--primary-background: 255 251 235;
--primary-text: 123 51 6;
--primary-accent: 0 166 244;
--secondary-background: 254 243 198;
--secondary-text: 151 60 0;
--secondary-accent: 79 57 246;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 255 251 235;
  --primary-background-hover: 242 238 223;
  --primary-background-focus: 230 226 212;
  --primary-background-disabled: 255 251 235;
  --primary-background-border: 209 206 193;
  --primary-background-gradient-a: 255 251 235;
  --primary-background-gradient-b: 247 243 227;
  --primary-background-gradient-c: 239 235 219;
  --primary-background-gradient-d: 217 213 197;

  --primary-text: 123 51 6;
  --primary-text-hover: 111 46 5;
  --primary-text-focus: 99 41 5;
  --primary-text-disabled: 157 101 70;
  --primary-text-border: 98 41 5;
  --primary-text-gradient-a: 123 51 6;
  --primary-text-gradient-b: 111 46 5;
  --primary-text-gradient-c: 99 41 5;
  --primary-text-gradient-d: 88 36 4;

  --primary-accent: 0 166 244;
  --primary-accent-hover: 0 158 232;
  --primary-accent-focus: 0 150 219;
  --primary-accent-disabled: 64 184 244;
  --primary-accent-border: 0 138 202;
  --primary-accent-gradient-a: 0 166 244;
  --primary-accent-gradient-b: 0 158 232;
  --primary-accent-gradient-c: 0 150 219;
  --primary-accent-gradient-d: 0 138 202;

  /* Secondary */
  --secondary-background: 254 243 198;
  --secondary-background-hover: 241 231 188;
  --secondary-background-focus: 229 219 178;
  --secondary-background-disabled: 254 243 198;
  --secondary-background-border: 208 199 162;
  --secondary-background-gradient-a: 254 243 198;
  --secondary-background-gradient-b: 246 235 190;
  --secondary-background-gradient-c: 238 227 182;
  --secondary-background-gradient-d: 216 206 165;

  --secondary-text: 151 60 0;
  --secondary-text-hover: 136 54 0;
  --secondary-text-focus: 121 48 0;
  --secondary-text-disabled: 178 110 40;
  --secondary-text-border: 121 48 0;
  --secondary-text-gradient-a: 151 60 0;
  --secondary-text-gradient-b: 136 54 0;
  --secondary-text-gradient-c: 121 48 0;
  --secondary-text-gradient-d: 106 42 0;

  --secondary-accent: 79 57 246;
  --secondary-accent-hover: 71 51 221;
  --secondary-accent-focus: 63 46 197;
  --secondary-accent-disabled: 120 103 246;
  --secondary-accent-border: 65 47 201;
  --secondary-accent-gradient-a: 79 57 246;
  --secondary-accent-gradient-b: 71 51 221;
  --secondary-accent-gradient-c: 63 46 197;
  --secondary-accent-gradient-d: 55 40 172;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 40 36 20;
  --primary-background-hover: 50 46 30;
  --primary-background-focus: 60 56 40;
  --primary-background-disabled: 33 29 16;
  --primary-background-border: 28 24 14;
  --primary-background-gradient-a: 40 36 20;
  --primary-background-gradient-b: 48 44 28;
  --primary-background-gradient-c: 56 52 36;
  --primary-background-gradient-d: 64 60 44;

  --primary-text: 245 215 170;
  --primary-text-hover: 250 225 180;
  --primary-text-focus: 255 235 190;
  --primary-text-disabled: 195 170 136;
  --primary-text-border: 225 195 150;
  --primary-text-gradient-a: 245 215 170;
  --primary-text-gradient-b: 247 225 180;
  --primary-text-gradient-c: 250 235 190;
  --primary-text-gradient-d: 253 245 200;

  --primary-accent: 0 188 255;
  --primary-accent-hover: 51 203 255;
  --primary-accent-focus: 102 218 255;
  --primary-accent-disabled: 0 151 204;
  --primary-accent-border: 0 169 230;
  --primary-accent-gradient-a: 0 188 255;
  --primary-accent-gradient-b: 40 198 255;
  --primary-accent-gradient-c: 80 208 255;
  --primary-accent-gradient-d: 120 218 255;

  /* Secondary */
  --secondary-background: 51 46 25;
  --secondary-background-hover: 61 56 35;
  --secondary-background-focus: 71 66 45;
  --secondary-background-disabled: 43 38 21;
  --secondary-background-border: 38 33 18;
  --secondary-background-gradient-a: 51 46 25;
  --secondary-background-gradient-b: 58 53 33;
  --secondary-background-gradient-c: 65 60 41;
  --secondary-background-gradient-d: 72 67 49;

  --secondary-text: 255 190 140;
  --secondary-text-hover: 255 200 150;
  --secondary-text-focus: 255 210 160;
  --secondary-text-disabled: 204 152 112;
  --secondary-text-border: 230 171 126;
  --secondary-text-gradient-a: 255 190 140;
  --secondary-text-gradient-b: 255 200 150;
  --secondary-text-gradient-c: 255 210 160;
  --secondary-text-gradient-d: 255 220 170;

  --secondary-accent: 120 100 255;
  --secondary-accent-hover: 140 120 255;
  --secondary-accent-focus: 160 140 255;
  --secondary-accent-disabled: 96 80 204;
  --secondary-accent-border: 108 90 230;
  --secondary-accent-gradient-a: 120 100 255;
  --secondary-accent-gradient-b: 135 115 255;
  --secondary-accent-gradient-c: 150 130 255;
  --secondary-accent-gradient-d: 165 145 255;
}
'''
}

colored_background = {
    "input": r'''--primary-background: 252 231 243;
--primary-text: 16 24 40;
--primary-accent: 230 0 118;
--secondary-background: 252 206 232;
--secondary-text: 30 41 57;
--secondary-accent: 198 0 92;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 252 231 243;
  --primary-background-hover: 239 219 231;
  --primary-background-focus: 227 208 219;
  --primary-background-disabled: 252 231 243;
  --primary-background-border: 209 192 202;
  --primary-background-gradient-a: 252 231 243;
  --primary-background-gradient-b: 245 225 237;
  --primary-background-gradient-c: 227 208 219;
  --primary-background-gradient-d: 207 189 199;

  --primary-text: 16 24 40;
  --primary-text-hover: 14 22 36;
  --primary-text-focus: 12 20 32;
  --primary-text-disabled: 74 80 92;
  --primary-text-border: 13 20 33;
  --primary-text-gradient-a: 16 24 40;
  --primary-text-gradient-b: 14 22 36;
  --primary-text-gradient-c: 12 20 32;
  --primary-text-gradient-d: 10 18 28;

  --primary-accent: 230 0 118;
  --primary-accent-hover: 219 0 112;
  --primary-accent-focus: 207 0 106;
  --primary-accent-disabled: 235 58 148;
  --primary-accent-border: 195 0 100;
  --primary-accent-gradient-a: 230 0 118;
  --primary-accent-gradient-b: 219 0 112;
  --primary-accent-gradient-c: 207 0 106;
  --primary-accent-gradient-d: 184 0 94;

  /* Secondary */
  --secondary-background: 252 206 232;
  --secondary-background-hover: 239 196 220;
  --secondary-background-focus: 227 185 209;
  --secondary-background-disabled: 252 206 232;
  --secondary-background-border: 209 171 192;
  --secondary-background-gradient-a: 252 206 232;
  --secondary-background-gradient-b: 245 201 226;
  --secondary-background-gradient-c: 227 185 209;
  --secondary-background-gradient-d: 207 169 190;

  --secondary-text: 30 41 57;
  --secondary-text-hover: 27 37 51;
  --secondary-text-focus: 24 33 46;
  --secondary-text-disabled: 84 92 104;
  --secondary-text-border: 25 34 47;
  --secondary-text-gradient-a: 30 41 57;
  --secondary-text-gradient-b: 27 37 51;
  --secondary-text-gradient-c: 24 33 46;
  --secondary-text-gradient-d: 21 29 40;

  --secondary-accent: 198 0 92;
  --secondary-accent-hover: 188 0 87;
  --secondary-accent-focus: 178 0 83;
  --secondary-accent-disabled: 209 50 123;
  --secondary-accent-border: 168 0 78;
  --secondary-accent-gradient-a: 198 0 92;
  --secondary-accent-gradient-b: 188 0 87;
  --secondary-accent-gradient-c: 178 0 83;
  --secondary-accent-gradient-d: 158 0 73;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 28 24 32;
  --primary-background-hover: 38 34 42;
  --primary-background-focus: 48 44 52;
  --primary-background-disabled: 23 19 27;
  --primary-background-border: 20 16 24;
  --primary-background-gradient-a: 28 24 32;
  --primary-background-gradient-b: 34 30 38;
  --primary-background-gradient-c: 40 36 44;
  --primary-background-gradient-d: 46 42 50;

  --primary-text: 230 225 240;
  --primary-text-hover: 240 235 250;
  --primary-text-focus: 250 245 255;
  --primary-text-disabled: 180 175 200;
  --primary-text-border: 210 205 220;
  --primary-text-gradient-a: 230 225 240;
  --primary-text-gradient-b: 237 232 245;
  --primary-text-gradient-c: 244 239 250;
  --primary-text-gradient-d: 251 246 255;

  --primary-accent: 255 50 145;
  --primary-accent-hover: 255 70 165;
  --primary-accent-focus: 255 90 185;
  --primary-accent-disabled: 204 40 116;
  --primary-accent-border: 229 45 130;
  --primary-accent-gradient-a: 255 50 145;
  --primary-accent-gradient-b: 255 65 160;
  --primary-accent-gradient-c: 255 80 175;
  --primary-accent-gradient-d: 255 95 190;

  /* Secondary */
  --secondary-background: 40 28 36;
  --secondary-background-hover: 50 38 46;
  --secondary-background-focus: 60 48 56;
  --secondary-background-disabled: 35 23 31;
  --secondary-background-border: 32 20 28;
  --secondary-background-gradient-a: 40 28 36;
  --secondary-background-gradient-b: 46 34 42;
  --secondary-background-gradient-c: 52 40 48;
  --secondary-background-gradient-d: 58 46 54;

  --secondary-text: 220 215 230;
  --secondary-text-hover: 230 225 240;
  --secondary-text-focus: 240 235 250;
  --secondary-text-disabled: 170 165 180;
  --secondary-text-border: 200 195 210;
  --secondary-text-gradient-a: 220 215 230;
  --secondary-text-gradient-b: 227 222 237;
  --secondary-text-gradient-c: 234 229 244;
  --secondary-text-gradient-d: 241 236 251;

  --secondary-accent: 233 40 121;
  --secondary-accent-hover: 243 60 141;
  --secondary-accent-focus: 253 80 161;
  --secondary-accent-disabled: 186 32 97;
  --secondary-accent-border: 210 36 109;
  --secondary-accent-gradient-a: 233 40 121;
  --secondary-accent-gradient-b: 239 53 134;
  --secondary-accent-gradient-c: 245 66 147;
  --secondary-accent-gradient-d: 251 79 160;
}
'''
}

dark_mode = {
    "input": r'''--primary-background: 16 78 100;
--primary-text: 236 254 255;
--primary-accent: 255 99 126;
--secondary-background: 0 95 120;
--secondary-text: 206 250 254;
--secondary-accent: 255 32 86;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 220 240 245;
  --primary-background-hover: 209 228 233;
  --primary-background-focus: 194 213 218;
  --primary-background-disabled: 220 240 245;
  --primary-background-border: 175 194 199;
  --primary-background-gradient-a: 220 240 245;
  --primary-background-gradient-b: 209 228 233;
  --primary-background-gradient-c: 194 213 218;
  --primary-background-gradient-d: 175 194 199;

  --primary-text: 16 78 100;
  --primary-text-hover: 14 70 90;
  --primary-text-focus: 12 62 80;
  --primary-text-disabled: 77 125 142;
  --primary-text-border: 10 54 72;
  --primary-text-gradient-a: 16 78 100;
  --primary-text-gradient-b: 14 70 90;
  --primary-text-gradient-c: 12 62 80;
  --primary-text-gradient-d: 10 54 72;

  --primary-accent: 255 99 126;
  --primary-accent-hover: 242 94 120;
  --primary-accent-focus: 229 89 114;
  --primary-accent-disabled: 255 138 156;
  --primary-accent-border: 204 79 101;
  --primary-accent-gradient-a: 255 99 126;
  --primary-accent-gradient-b: 242 94 120;
  --primary-accent-gradient-c: 229 89 114;
  --primary-accent-gradient-d: 204 79 101;

  /* Secondary */
  --secondary-background: 210 235 242;
  --secondary-background-hover: 200 223 230;
  --secondary-background-focus: 189 212 218;
  --secondary-background-disabled: 210 235 242;
  --secondary-background-border: 172 193 198;
  --secondary-background-gradient-a: 210 235 242;
  --secondary-background-gradient-b: 200 223 230;
  --secondary-background-gradient-c: 189 212 218;
  --secondary-background-gradient-d: 172 193 198;

  --secondary-text: 0 95 120;
  --secondary-text-hover: 0 85 108;
  --secondary-text-focus: 0 75 96;
  --secondary-text-disabled: 66 137 156;
  --secondary-text-border: 0 65 84;
  --secondary-text-gradient-a: 0 95 120;
  --secondary-text-gradient-b: 0 85 108;
  --secondary-text-gradient-c: 0 75 96;
  --secondary-text-gradient-d: 0 65 84;

  --secondary-accent: 255 32 86;
  --secondary-accent-hover: 242 30 82;
  --secondary-accent-focus: 229 28 77;
  --secondary-accent-disabled: 255 83 124;
  --secondary-accent-border: 204 26 69;
  --secondary-accent-gradient-a: 255 32 86;
  --secondary-accent-gradient-b: 242 30 82;
  --secondary-accent-gradient-c: 229 28 77;
  --secondary-accent-gradient-d: 204 26 69;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 16 78 100;
  --primary-background-hover: 19 93 120;
  --primary-background-focus: 22 109 140;
  --primary-background-disabled: 13 62 80;
  --primary-background-border: 12 58 75;
  --primary-background-gradient-a: 16 78 100;
  --primary-background-gradient-b: 19 93 120;
  --primary-background-gradient-c: 22 109 140;
  --primary-background-gradient-d: 25 124 160;

  --primary-text: 236 254 255;
  --primary-text-hover: 245 255 255;
  --primary-text-focus: 255 255 255;
  --primary-text-disabled: 189 203 204;
  --primary-text-border: 224 241 242;
  --primary-text-gradient-a: 236 254 255;
  --primary-text-gradient-b: 241 255 255;
  --primary-text-gradient-c: 246 255 255;
  --primary-text-gradient-d: 255 255 255;

  --primary-accent: 255 99 126;
  --primary-accent-hover: 255 119 146;
  --primary-accent-focus: 255 139 166;
  --primary-accent-disabled: 204 79 101;
  --primary-accent-border: 229 89 113;
  --primary-accent-gradient-a: 255 99 126;
  --primary-accent-gradient-b: 255 119 146;
  --primary-accent-gradient-c: 255 139 166;
  --primary-accent-gradient-d: 255 159 186;

  /* Secondary */
  --secondary-background: 0 95 120;
  --secondary-background-hover: 0 113 143;
  --secondary-background-focus: 0 132 167;
  --secondary-background-disabled: 0 76 96;
  --secondary-background-border: 0 71 90;
  --secondary-background-gradient-a: 0 95 120;
  --secondary-background-gradient-b: 0 113 143;
  --secondary-background-gradient-c: 0 132 167;
  --secondary-background-gradient-d: 0 150 190;

  --secondary-text: 206 250 254;
  --secondary-text-hover: 216 252 255;
  --secondary-text-focus: 226 254 255;
  --secondary-text-disabled: 165 200 203;
  --secondary-text-border: 196 238 242;
  --secondary-text-gradient-a: 206 250 254;
  --secondary-text-gradient-b: 216 252 255;
  --secondary-text-gradient-c: 226 254 255;
  --secondary-text-gradient-d: 236 255 255;

  --secondary-accent: 255 32 86;
  --secondary-accent-hover: 255 52 106;
  --secondary-accent-focus: 255 72 126;
  --secondary-accent-disabled: 204 26 69;
  --secondary-accent-border: 229 29 77;
  --secondary-accent-gradient-a: 255 32 86;
  --secondary-accent-gradient-b: 255 52 106;
  --secondary-accent-gradient-c: 255 72 126;
  --secondary-accent-gradient-d: 255 92 146;
}
'''
}

pastel = {
    "input": r'''--primary-background: 243 232 255;
--primary-text: 30 41 57;
--primary-accent: 123 241 168;
--secondary-background: 250 245 255;
--secondary-text: 54 65 83;
--secondary-accent: 5 223 114;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 243 232 255;
  --primary-background-hover: 231 220 242;
  --primary-background-focus: 219 209 230;
  --primary-background-disabled: 243 232 255;
  --primary-background-border: 199 190 209;
  --primary-background-gradient-a: 243 232 255;
  --primary-background-gradient-b: 234 223 246;
  --primary-background-gradient-c: 225 214 237;
  --primary-background-gradient-d: 206 196 217;

  --primary-text: 30 41 57;
  --primary-text-hover: 28 39 54;
  --primary-text-focus: 27 37 51;
  --primary-text-disabled: 74 85 101;
  --primary-text-border: 25 34 47;
  --primary-text-gradient-a: 30 41 57;
  --primary-text-gradient-b: 28 39 54;
  --primary-text-gradient-c: 27 37 51;
  --primary-text-gradient-d: 25 34 47;

  --primary-accent: 123 241 168;
  --primary-accent-hover: 117 229 160;
  --primary-accent-focus: 111 217 151;
  --primary-accent-disabled: 148 244 184;
  --primary-accent-border: 101 198 138;
  --primary-accent-gradient-a: 123 241 168;
  --primary-accent-gradient-b: 117 229 160;
  --primary-accent-gradient-c: 111 217 151;
  --primary-accent-gradient-d: 101 198 138;

  /* Secondary */
  --secondary-background: 250 245 255;
  --secondary-background-hover: 238 233 242;
  --secondary-background-focus: 225 221 230;
  --secondary-background-disabled: 250 245 255;
  --secondary-background-border: 205 201 209;
  --secondary-background-gradient-a: 250 245 255;
  --secondary-background-gradient-b: 241 236 246;
  --secondary-background-gradient-c: 232 227 237;
  --secondary-background-gradient-d: 213 208 217;

  --secondary-text: 54 65 83;
  --secondary-text-hover: 51 62 79;
  --secondary-text-focus: 49 59 75;
  --secondary-text-disabled: 98 109 127;
  --secondary-text-border: 44 53 68;
  --secondary-text-gradient-a: 54 65 83;
  --secondary-text-gradient-b: 51 62 79;
  --secondary-text-gradient-c: 49 59 75;
  --secondary-text-gradient-d: 44 53 68;

  --secondary-accent: 5 223 114;
  --secondary-accent-hover: 5 212 108;
  --secondary-accent-focus: 4 201 103;
  --secondary-accent-disabled: 55 229 138;
  --secondary-accent-border: 4 183 93;
  --secondary-accent-gradient-a: 5 223 114;
  --secondary-accent-gradient-b: 5 212 108;
  --secondary-accent-gradient-c: 4 201 103;
  --secondary-accent-gradient-d: 4 183 93;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 30 23 46;
  --primary-background-hover: 40 33 56;
  --primary-background-focus: 50 43 66;
  --primary-background-disabled: 25 18 41;
  --primary-background-border: 22 15 38;
  --primary-background-gradient-a: 30 23 46;
  --primary-background-gradient-b: 37 30 53;
  --primary-background-gradient-c: 44 37 60;
  --primary-background-gradient-d: 51 44 67;

  --primary-text: 220 225 235;
  --primary-text-hover: 230 235 245;
  --primary-text-focus: 240 245 255;
  --primary-text-disabled: 180 185 195;
  --primary-text-border: 209 214 224;
  --primary-text-gradient-a: 220 225 235;
  --primary-text-gradient-b: 228 233 243;
  --primary-text-gradient-c: 236 241 251;
  --primary-text-gradient-d: 244 249 255;

  --primary-accent: 143 255 188;
  --primary-accent-hover: 153 255 198;
  --primary-accent-focus: 163 255 208;
  --primary-accent-disabled: 114 204 150;
  --primary-accent-border: 129 230 169;
  --primary-accent-gradient-a: 143 255 188;
  --primary-accent-gradient-b: 153 255 198;
  --primary-accent-gradient-c: 163 255 208;
  --primary-accent-gradient-d: 173 255 218;

  /* Secondary */
  --secondary-background: 37 30 53;
  --secondary-background-hover: 47 40 63;
  --secondary-background-focus: 57 50 73;
  --secondary-background-disabled: 32 25 48;
  --secondary-background-border: 29 22 45;
  --secondary-background-gradient-a: 37 30 53;
  --secondary-background-gradient-b: 44 37 60;
  --secondary-background-gradient-c: 51 44 67;
  --secondary-background-gradient-d: 58 51 74;

  --secondary-text: 200 205 215;
  --secondary-text-hover: 210 215 225;
  --secondary-text-focus: 220 225 235;
  --secondary-text-disabled: 160 165 175;
  --secondary-text-border: 190 195 205;
  --secondary-text-gradient-a: 200 205 215;
  --secondary-text-gradient-b: 208 213 223;
  --secondary-text-gradient-c: 216 221 231;
  --secondary-text-gradient-d: 224 229 239;

  --secondary-accent: 25 243 134;
  --secondary-accent-hover: 35 253 144;
  --secondary-accent-focus: 45 255 154;
  --secondary-accent-disabled: 20 194 107;
  --secondary-accent-border: 23 219 121;
  --secondary-accent-gradient-a: 25 243 134;
  --secondary-accent-gradient-b: 35 249 141;
  --secondary-accent-gradient-c: 45 255 148;
  --secondary-accent-gradient-d: 55 255 155;
}
'''
}

corporate = {
    "input": r'''--primary-background: 249 250 251;
--primary-text: 16 24 40;
--primary-accent: 0 150 137;
--secondary-background: 243 244 246;
--secondary-text: 30 41 57;
--secondary-accent: 0 187 167;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 249 250 251;
  --primary-background-hover: 237 238 239;
  --primary-background-focus: 219 220 221;
  --primary-background-disabled: 249 250 251;
  --primary-background-border: 199 200 201;
  --primary-background-gradient-a: 249 250 251;
  --primary-background-gradient-b: 239 240 241;
  --primary-background-gradient-c: 229 230 231;
  --primary-background-gradient-d: 209 210 211;

  --primary-text: 16 24 40;
  --primary-text-hover: 14 22 38;
  --primary-text-focus: 12 20 36;
  --primary-text-disabled: 86 94 110;
  --primary-text-border: 13 20 34;
  --primary-text-gradient-a: 16 24 40;
  --primary-text-gradient-b: 14 22 38;
  --primary-text-gradient-c: 12 20 36;
  --primary-text-gradient-d: 10 18 34;

  --primary-accent: 0 150 137;
  --primary-accent-hover: 0 140 128;
  --primary-accent-focus: 0 130 119;
  --primary-accent-disabled: 75 180 170;
  --primary-accent-border: 0 125 114;
  --primary-accent-gradient-a: 0 150 137;
  --primary-accent-gradient-b: 0 140 128;
  --primary-accent-gradient-c: 0 130 119;
  --primary-accent-gradient-d: 0 120 110;

  /* Secondary */
  --secondary-background: 243 244 246;
  --secondary-background-hover: 231 232 234;
  --secondary-background-focus: 219 220 222;
  --secondary-background-disabled: 243 244 246;
  --secondary-background-border: 206 207 209;
  --secondary-background-gradient-a: 243 244 246;
  --secondary-background-gradient-b: 233 234 236;
  --secondary-background-gradient-c: 223 224 226;
  --secondary-background-gradient-d: 203 204 206;

  --secondary-text: 30 41 57;
  --secondary-text-hover: 28 39 54;
  --secondary-text-focus: 26 37 51;
  --secondary-text-disabled: 95 106 122;
  --secondary-text-border: 25 35 48;
  --secondary-text-gradient-a: 30 41 57;
  --secondary-text-gradient-b: 28 39 54;
  --secondary-text-gradient-c: 26 37 51;
  --secondary-text-gradient-d: 24 35 48;

  --secondary-accent: 0 187 167;
  --secondary-accent-hover: 0 175 157;
  --secondary-accent-focus: 0 165 147;
  --secondary-accent-disabled: 80 210 193;
  --secondary-accent-border: 0 155 138;
  --secondary-accent-gradient-a: 0 187 167;
  --secondary-accent-gradient-b: 0 175 157;
  --secondary-accent-gradient-c: 0 165 147;
  --secondary-accent-gradient-d: 0 155 138;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 30 32 35;
  --primary-background-hover: 40 42 45;
  --primary-background-focus: 50 52 55;
  --primary-background-disabled: 25 27 30;
  --primary-background-border: 22 24 27;
  --primary-background-gradient-a: 30 32 35;
  --primary-background-gradient-b: 37 39 42;
  --primary-background-gradient-c: 44 46 49;
  --primary-background-gradient-d: 51 53 56;

  --primary-text: 230 235 245;
  --primary-text-hover: 240 245 255;
  --primary-text-focus: 250 252 255;
  --primary-text-disabled: 190 195 205;
  --primary-text-border: 215 220 230;
  --primary-text-gradient-a: 230 235 245;
  --primary-text-gradient-b: 237 242 250;
  --primary-text-gradient-c: 244 249 255;
  --primary-text-gradient-d: 251 253 255;

  --primary-accent: 0 180 167;
  --primary-accent-hover: 0 195 182;
  --primary-accent-focus: 0 210 197;
  --primary-accent-disabled: 0 145 135;
  --primary-accent-border: 0 165 153;
  --primary-accent-gradient-a: 0 180 167;
  --primary-accent-gradient-b: 0 190 177;
  --primary-accent-gradient-c: 0 200 187;
  --primary-accent-gradient-d: 0 210 197;

  /* Secondary */
  --secondary-background: 40 42 47;
  --secondary-background-hover: 50 52 57;
  --secondary-background-focus: 60 62 67;
  --secondary-background-disabled: 35 37 42;
  --secondary-background-border: 32 34 39;
  --secondary-background-gradient-a: 40 42 47;
  --secondary-background-gradient-b: 47 49 54;
  --secondary-background-gradient-c: 54 56 61;
  --secondary-background-gradient-d: 61 63 68;

  --secondary-text: 220 228 240;
  --secondary-text-hover: 230 238 250;
  --secondary-text-focus: 240 248 255;
  --secondary-text-disabled: 180 188 200;
  --secondary-text-border: 205 213 225;
  --secondary-text-gradient-a: 220 228 240;
  --secondary-text-gradient-b: 227 235 247;
  --secondary-text-gradient-c: 234 242 250;
  --secondary-text-gradient-d: 241 249 255;

  --secondary-accent: 0 217 197;
  --secondary-accent-hover: 0 232 212;
  --secondary-accent-focus: 0 247 227;
  --secondary-accent-disabled: 0 180 162;
  --secondary-accent-border: 0 200 182;
  --secondary-accent-gradient-a: 0 217 197;
  --secondary-accent-gradient-b: 0 227 207;
  --secondary-accent-gradient-c: 0 237 217;
  --secondary-accent-gradient-d: 0 247 227;
}
'''
}

vibrant = {
    "input": r'''--primary-background: 243 232 255;
--primary-text: 16 24 40;
--primary-accent: 240 177 0;
--secondary-background: 233 212 255;
--secondary-text: 30 41 57;
--secondary-accent: 208 135 0;
''',
  "output": r'''/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: 243 232 255;
  --primary-background-hover: 231 220 243;
  --primary-background-focus: 219 208 231;
  --primary-background-disabled: 243 232 255;
  --primary-background-border: 206 196 217;
  --primary-background-gradient-a: 243 232 255;
  --primary-background-gradient-b: 234 223 246;
  --primary-background-gradient-c: 225 214 237;
  --primary-background-gradient-d: 206 196 217;

  --primary-text: 16 24 40;
  --primary-text-hover: 12 20 36;
  --primary-text-focus: 8 16 32;
  --primary-text-disabled: 61 69 85;
  --primary-text-border: 13 20 34;
  --primary-text-gradient-a: 16 24 40;
  --primary-text-gradient-b: 13 21 37;
  --primary-text-gradient-c: 10 18 34;
  --primary-text-gradient-d: 7 15 31;

  --primary-accent: 240 177 0;
  --primary-accent-hover: 228 168 0;
  --primary-accent-focus: 216 159 0;
  --primary-accent-disabled: 241 191 51;
  --primary-accent-border: 204 150 0;
  --primary-accent-gradient-a: 240 177 0;
  --primary-accent-gradient-b: 228 168 0;
  --primary-accent-gradient-c: 216 159 0;
  --primary-accent-gradient-d: 204 150 0;

  /* Secondary */
  --secondary-background: 233 212 255;
  --secondary-background-hover: 221 201 242;
  --secondary-background-focus: 209 190 229;
  --secondary-background-disabled: 233 212 255;
  --secondary-background-border: 198 180 217;
  --secondary-background-gradient-a: 233 212 255;
  --secondary-background-gradient-b: 224 203 246;
  --secondary-background-gradient-c: 215 194 237;
  --secondary-background-gradient-d: 198 180 217;

  --secondary-text: 30 41 57;
  --secondary-text-hover: 26 37 53;
  --secondary-text-focus: 22 33 49;
  --secondary-text-disabled: 76 87 103;
  --secondary-text-border: 25 35 48;
  --secondary-text-gradient-a: 30 41 57;
  --secondary-text-gradient-b: 27 38 54;
  --secondary-text-gradient-c: 24 35 51;
  --secondary-text-gradient-d: 21 32 48;

  --secondary-accent: 208 135 0;
  --secondary-accent-hover: 197 128 0;
  --secondary-accent-focus: 187 121 0;
  --secondary-accent-disabled: 214 156 51;
  --secondary-accent-border: 177 115 0;
  --secondary-accent-gradient-a: 208 135 0;
  --secondary-accent-gradient-b: 197 128 0;
  --secondary-accent-gradient-c: 187 121 0;
  --secondary-accent-gradient-d: 177 115 0;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: 45 34 67;
  --primary-background-hover: 55 44 77;
  --primary-background-focus: 65 54 87;
  --primary-background-disabled: 35 24 57;
  --primary-background-border: 30 19 52;
  --primary-background-gradient-a: 45 34 67;
  --primary-background-gradient-b: 55 44 77;
  --primary-background-gradient-c: 65 54 87;
  --primary-background-gradient-d: 75 64 97;

  --primary-text: 225 220 240;
  --primary-text-hover: 235 230 250;
  --primary-text-focus: 245 240 255;
  --primary-text-disabled: 175 170 190;
  --primary-text-border: 205 200 220;
  --primary-text-gradient-a: 225 220 240;
  --primary-text-gradient-b: 235 230 250;
  --primary-text-gradient-c: 245 240 255;
  --primary-text-gradient-d: 255 250 255;

  --primary-accent: 255 192 20;
  --primary-accent-hover: 255 202 40;
  --primary-accent-focus: 255 212 60;
  --primary-accent-disabled: 204 153 16;
  --primary-accent-border: 230 173 18;
  --primary-accent-gradient-a: 255 192 20;
  --primary-accent-gradient-b: 255 202 40;
  --primary-accent-gradient-c: 255 212 60;
  --primary-accent-gradient-d: 255 222 80;

  /* Secondary */
  --secondary-background: 57 36 79;
  --secondary-background-hover: 67 46 89;
  --secondary-background-focus: 77 56 99;
  --secondary-background-disabled: 47 26 69;
  --secondary-background-border: 42 21 64;
  --secondary-background-gradient-a: 57 36 79;
  --secondary-background-gradient-b: 67 46 89;
  --secondary-background-gradient-c: 77 56 99;
  --secondary-background-gradient-d: 87 66 109;

  --secondary-text: 210 205 225;
  --secondary-text-hover: 220 215 235;
  --secondary-text-focus: 230 225 245;
  --secondary-text-disabled: 160 155 175;
  --secondary-text-border: 190 185 205;
  --secondary-text-gradient-a: 210 205 225;
  --secondary-text-gradient-b: 220 215 235;
  --secondary-text-gradient-c: 230 225 245;
  --secondary-text-gradient-d: 240 235 255;

  --secondary-accent: 228 155 20;
  --secondary-accent-hover: 238 165 40;
  --secondary-accent-focus: 248 175 60;
  --secondary-accent-disabled: 182 124 16;
  --secondary-accent-border: 208 140 18;
  --secondary-accent-gradient-a: 228 155 20;
  --secondary-accent-gradient-b: 238 165 40;
  --secondary-accent-gradient-c: 248 175 60;
  --secondary-accent-gradient-d: 255 185 80;
}
'''
}
