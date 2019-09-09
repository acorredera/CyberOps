import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { FiltroPalabrasPage } from './filtro-palabras.page';

describe('FiltroPalabrasPage', () => {
  let component: FiltroPalabrasPage;
  let fixture: ComponentFixture<FiltroPalabrasPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ FiltroPalabrasPage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(FiltroPalabrasPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
